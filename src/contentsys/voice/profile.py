"""Building and storing voice profiles.

One profile per platform, because register genuinely differs. The same person
writes lowercase fragments on X and structured paragraphs on LinkedIn, and
averaging those produces a voice that matches neither.

Profiles are versioned rather than overwritten, so a regression in generated
voice can be traced to the profile that caused it.
"""

from __future__ import annotations

from sqlmodel import Session, select

from contentsys.config import Platform
from contentsys.db.models import SampleSource, VoiceProfile, WritingSample
from contentsys.voice.surface import SurfaceProfile, analyse

#: Which sample sources inform which platform's voice.
#:
#: Medium feeds the LinkedIn profile because both are long form explanatory
#: writing by the same person. It deliberately does not feed X, where the
#: register is completely different.
PLATFORM_SOURCES: dict[Platform, tuple[SampleSource, ...]] = {
    Platform.X: (SampleSource.X,),
    Platform.LINKEDIN: (SampleSource.LINKEDIN, SampleSource.MEDIUM),
}

#: Below this, the measurements are noise. The profile is still built, because
#: something is better than nothing on day one, but the caller is warned.
MIN_USEFUL_SAMPLES = 15


def samples_for(session: Session, platform: Platform) -> list[WritingSample]:
    """Every usable sample for a platform, excluding the ones marked out."""
    sources = PLATFORM_SOURCES[platform]
    statement = select(WritingSample).where(
        WritingSample.source.in_(sources),  # type: ignore[attr-defined]
        WritingSample.excluded == False,  # noqa: E712 - SQL comparison, not Python
    )
    return list(session.exec(statement))


def build_profile(session: Session, platform: Platform) -> tuple[VoiceProfile, SurfaceProfile]:
    """Measure the owner's writing and store a new profile version.

    The semantic half is left empty here. It needs a model call and lands with
    the generation phase; the surface half is deterministic and useful on its
    own, both as a sanity check on ingestion and as a post-generation gate.
    """
    samples = samples_for(session, platform)
    surface = analyse([sample.content for sample in samples])

    previous = session.exec(
        select(VoiceProfile)
        .where(VoiceProfile.platform == platform)
        .order_by(VoiceProfile.version.desc())  # type: ignore[attr-defined]
    ).first()

    for existing in session.exec(
        select(VoiceProfile).where(
            VoiceProfile.platform == platform,
            VoiceProfile.is_active == True,  # noqa: E712 - SQL comparison
        )
    ):
        existing.is_active = False
        session.add(existing)

    profile = VoiceProfile(
        platform=platform,
        version=(previous.version + 1) if previous else 1,
        is_active=True,
        sample_count=len(samples),
        surface=surface.to_dict(),
        semantic=dict(previous.semantic) if previous else {},
    )
    session.add(profile)
    return profile, surface


def active_profile(session: Session, platform: Platform) -> VoiceProfile | None:
    return session.exec(
        select(VoiceProfile).where(
            VoiceProfile.platform == platform,
            VoiceProfile.is_active == True,  # noqa: E712 - SQL comparison
        )
    ).first()


def load_surface(profile: VoiceProfile) -> SurfaceProfile:
    """Rebuild the dataclass from the stored JSON.

    Tolerates fields added since the profile was written, so an old profile
    stays readable after the analyser grows a new measurement.
    """
    known = {field for field in SurfaceProfile.__dataclass_fields__}
    return SurfaceProfile(**{k: v for k, v in profile.surface.items() if k in known})
