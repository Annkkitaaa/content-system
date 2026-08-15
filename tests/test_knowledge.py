from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlmodel import Session, select

from contentsys.config import Platform
from contentsys.db.models import (
    Confidence,
    Experience,
    KnowledgeItem,
    Opinion,
    SampleSource,
    Topic,
    VoiceProfile,
    WritingSample,
)
from contentsys.knowledge import add_sample, fingerprint, import_samples, load_seed
from contentsys.voice import build_profile, load_surface, samples_for

SEED_DIR = Path(__file__).resolve().parent.parent / "seed"


class TestFingerprint:
    def test_identical_text_matches(self) -> None:
        assert fingerprint("hello world") == fingerprint("hello world")

    def test_whitespace_and_case_are_normalised(self) -> None:
        # A re-export with different line wrapping is the same post.
        assert fingerprint("Hello   World") == fingerprint("hello world")
        assert fingerprint("hello\nworld") == fingerprint("hello world")

    def test_different_text_does_not_collide(self) -> None:
        assert fingerprint("hello world") != fingerprint("hello there")


class TestAddSample:
    def test_stores_a_sample(self, session: Session) -> None:
        sample = add_sample(session, content="i think this is right", source=SampleSource.X)

        assert sample is not None
        assert sample.source is SampleSource.X

    def test_rejects_a_duplicate(self, session: Session) -> None:
        add_sample(session, content="same text", source=SampleSource.X)
        session.commit()

        assert add_sample(session, content="  Same   Text  ", source=SampleSource.X) is None

    def test_rejects_text_below_the_minimum(self, session: Session) -> None:
        assert add_sample(session, content="a", source=SampleSource.X, min_length=5) is None


class TestImportFormats:
    def test_plain_text_splits_on_blank_lines(self, session: Session, tmp_path: Path) -> None:
        path = tmp_path / "posts.txt"
        path.write_text("first post\n\nsecond post\n\nthird post", encoding="utf-8")

        report = import_samples(session, path, SampleSource.X)

        assert report.added == 3

    def test_csv_finds_the_text_column(self, session: Session, tmp_path: Path) -> None:
        path = tmp_path / "posts.csv"
        path.write_text(
            "date,text,impressions\n2026-08-11,a real post,1200\n2026-08-10,another one,900\n",
            encoding="utf-8",
        )

        report = import_samples(session, path, SampleSource.X)
        stored = session.exec(select(WritingSample)).all()

        assert report.added == 2
        assert {sample.impressions for sample in stored} == {1200, 900}
        assert all(sample.published_at is not None for sample in stored)

    def test_x_archive_json_is_understood(self, session: Session, tmp_path: Path) -> None:
        # The archive wraps its JSON in a JavaScript assignment and nests each
        # post under "tweet". Both are handled so an export works untouched.
        path = tmp_path / "tweets.js"
        path.write_text(
            "window.YTD.tweets.part0 = "
            + json.dumps(
                [
                    {"tweet": {"full_text": "a post from the archive", "created_at": "2026-08-11"}},
                    {"tweet": {"full_text": "another archived post"}},
                ]
            ),
            encoding="utf-8",
        )
        path = path.rename(tmp_path / "tweets.json")

        report = import_samples(session, path, SampleSource.X)

        assert report.added == 2

    def test_reimporting_the_same_export_adds_nothing(
        self, session: Session, tmp_path: Path
    ) -> None:
        # Duplicates quietly bias the voice measurement, so this matters more
        # than it looks.
        path = tmp_path / "posts.txt"
        path.write_text("first post\n\nsecond post", encoding="utf-8")
        import_samples(session, path, SampleSource.X)
        session.commit()

        second = import_samples(session, path, SampleSource.X)

        assert second.added == 0
        assert second.skipped_duplicate == 2

    def test_report_reads_clearly(self, session: Session, tmp_path: Path) -> None:
        path = tmp_path / "posts.txt"
        path.write_text("only post", encoding="utf-8")

        assert "1 added" in import_samples(session, path, SampleSource.X).describe()


class TestSeed:
    @pytest.fixture
    def seeded(self, session: Session) -> Session:
        load_seed(session, SEED_DIR)
        session.commit()
        return session

    def test_loads_the_committed_starting_data(self, seeded: Session) -> None:
        assert len(seeded.exec(select(WritingSample)).all()) > 20
        assert len(seeded.exec(select(Experience)).all()) > 5
        assert len(seeded.exec(select(Opinion)).all()) > 5
        assert len(seeded.exec(select(KnowledgeItem)).all()) > 10
        assert len(seeded.exec(select(Topic)).all()) > 5

    def test_running_twice_changes_nothing(self, seeded: Session) -> None:
        before = len(seeded.exec(select(WritingSample)).all())

        reports = load_seed(seeded, SEED_DIR)
        seeded.commit()

        assert len(seeded.exec(select(WritingSample)).all()) == before
        assert reports["samples"].added == 0

    def test_unconfirmed_experiences_cannot_back_a_first_person_claim(
        self, seeded: Session
    ) -> None:
        # The invariant, at the data layer. An inferred experience is real
        # information but not permission to write "when I was working on".
        inferred = seeded.exec(
            select(Experience).where(Experience.confidence == Confidence.INFERRED)
        ).all()

        assert inferred, "expected at least one unconfirmed experience in the seed"
        assert all(not item.is_usable_for_first_person for item in inferred)

    def test_confirmed_experiences_carry_evidence(self, seeded: Session) -> None:
        # A claim you cannot trace back to a source is a claim you cannot defend.
        confirmed = seeded.exec(
            select(Experience).where(Experience.confidence == Confidence.STATED)
        ).all()

        assert all(item.evidence or item.evidence_url for item in confirmed)

    def test_excluded_samples_stay_out_of_the_voice_measurement(self, seeded: Session) -> None:
        excluded = seeded.exec(
            select(WritingSample).where(WritingSample.excluded == True)  # noqa: E712
        ).all()
        usable = samples_for(seeded, Platform.X)

        assert excluded
        assert all(sample.exclusion_reason for sample in excluded)
        assert not any(sample.excluded for sample in usable)


class TestVoiceProfileBuilding:
    @pytest.fixture
    def seeded(self, session: Session) -> Session:
        load_seed(session, SEED_DIR)
        session.commit()
        return session

    def test_builds_a_profile_per_platform(self, seeded: Session) -> None:
        x_profile, _ = build_profile(seeded, Platform.X)
        li_profile, _ = build_profile(seeded, Platform.LINKEDIN)
        seeded.commit()

        assert x_profile.sample_count > 0
        assert li_profile.sample_count > 0
        assert x_profile.version == 1

    def test_the_measured_profile_matches_the_real_voice(self, seeded: Session) -> None:
        # The point of the whole phase. If this drifts, generated posts will
        # sound like someone else and nothing downstream will catch it.
        _, surface = build_profile(seeded, Platform.X)

        assert surface.all_lowercase_post_ratio > 0.6, "this voice writes in lowercase"
        assert surface.emoji_ratio == 0.0, "this voice does not use emoji"
        assert surface.median_sentence_words < 15, "this voice writes short sentences"
        assert surface.mean_sentences_per_post < 4, "these are short posts"

    def test_registers_are_kept_apart(self, seeded: Session) -> None:
        # Averaging a lowercase X voice with structured LinkedIn prose gives a
        # voice that matches neither.
        _, x_surface = build_profile(seeded, Platform.X)
        _, li_surface = build_profile(seeded, Platform.LINKEDIN)

        assert li_surface.mean_sentences_per_post > x_surface.mean_sentences_per_post

    def test_rebuilding_supersedes_the_previous_version(self, seeded: Session) -> None:
        build_profile(seeded, Platform.X)
        seeded.commit()

        second, _ = build_profile(seeded, Platform.X)
        seeded.commit()

        active = seeded.exec(
            select(VoiceProfile).where(
                VoiceProfile.platform == Platform.X,
                VoiceProfile.is_active == True,  # noqa: E712
            )
        ).all()

        assert second.version == 2
        assert len(active) == 1, "exactly one profile should be active per platform"

    def test_a_stored_profile_survives_a_round_trip(self, seeded: Session) -> None:
        profile, surface = build_profile(seeded, Platform.X)
        seeded.commit()

        assert load_surface(profile).median_sentence_words == surface.median_sentence_words

    def test_an_old_profile_still_loads_after_a_new_metric_is_added(self, seeded: Session) -> None:
        # Forward compatibility: adding a measurement must not break profiles
        # written before it existed.
        profile, _ = build_profile(seeded, Platform.X)
        profile.surface["a_metric_from_the_future"] = 1.0

        assert load_surface(profile).sample_count == profile.sample_count

    def test_a_platform_with_no_samples_still_builds(self, session: Session) -> None:
        profile, surface = build_profile(session, Platform.X)

        assert profile.sample_count == 0
        assert surface.sample_count == 0
