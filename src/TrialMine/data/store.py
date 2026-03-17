"""SQLite persistence layer for Trial objects.

Schema mirrors the Trial Pydantic model.
List fields (conditions, interventions, locations) are stored as JSON text.
Indexes on nct_id, status, phase for fast filtering.
"""

import json
import logging
from pathlib import Path

from sqlalchemy import Index, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from TrialMine.data.models import Location, Trial

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class TrialRow(Base):
    """SQLAlchemy ORM model for the trials table."""

    __tablename__ = "trials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nct_id: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    title: Mapped[str] = mapped_column(Text, default="")
    brief_summary: Mapped[str | None] = mapped_column(Text)
    detailed_description: Mapped[str | None] = mapped_column(Text)
    conditions: Mapped[str] = mapped_column(Text, default="[]")       # JSON list
    interventions: Mapped[str] = mapped_column(Text, default="[]")    # JSON list
    eligibility_criteria: Mapped[str | None] = mapped_column(Text)
    min_age: Mapped[str | None] = mapped_column(String(50))
    max_age: Mapped[str | None] = mapped_column(String(50))
    sex: Mapped[str | None] = mapped_column(String(10))
    phase: Mapped[str | None] = mapped_column(String(50))
    status: Mapped[str | None] = mapped_column(String(50))
    enrollment: Mapped[int | None] = mapped_column(Integer)
    start_date: Mapped[str | None] = mapped_column(String(20))
    completion_date: Mapped[str | None] = mapped_column(String(20))
    sponsor: Mapped[str | None] = mapped_column(Text)
    locations: Mapped[str] = mapped_column(Text, default="[]")        # JSON list of dicts
    url: Mapped[str | None] = mapped_column(String(100))


# Explicit indexes for common filter/lookup patterns
_idx_nct_id = Index("ix_trials_nct_id", TrialRow.nct_id)
_idx_status = Index("ix_trials_status", TrialRow.status)
_idx_phase = Index("ix_trials_phase", TrialRow.phase)


def _to_row(trial: Trial) -> TrialRow:
    return TrialRow(
        nct_id=trial.nct_id,
        title=trial.title,
        brief_summary=trial.brief_summary,
        detailed_description=trial.detailed_description,
        conditions=json.dumps(trial.conditions),
        interventions=json.dumps(trial.interventions),
        eligibility_criteria=trial.eligibility_criteria,
        min_age=trial.min_age,
        max_age=trial.max_age,
        sex=trial.sex,
        phase=trial.phase,
        status=trial.status,
        enrollment=trial.enrollment,
        start_date=trial.start_date,
        completion_date=trial.completion_date,
        sponsor=trial.sponsor,
        locations=json.dumps([loc.model_dump() for loc in trial.locations]),
        url=trial.url,
    )


def _from_row(row: TrialRow) -> Trial:
    raw_locations = json.loads(row.locations or "[]")
    return Trial(
        nct_id=row.nct_id,
        title=row.title or "",
        brief_summary=row.brief_summary,
        detailed_description=row.detailed_description,
        conditions=json.loads(row.conditions or "[]"),
        interventions=json.loads(row.interventions or "[]"),
        eligibility_criteria=row.eligibility_criteria,
        min_age=row.min_age,
        max_age=row.max_age,
        sex=row.sex,
        phase=row.phase,
        status=row.status,
        enrollment=row.enrollment,
        start_date=row.start_date,
        completion_date=row.completion_date,
        sponsor=row.sponsor,
        locations=[Location(**loc) for loc in raw_locations],
        url=row.url,
    )


def get_engine(db_path: Path):
    """Create a SQLAlchemy engine for the given SQLite path."""
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db(db_path: Path) -> None:
    """Create the database and all tables if they don't exist.

    Args:
        db_path: Path to the SQLite database file.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    logger.info("Database initialised at %s", db_path)


def store_trials(trials: list[Trial], db_path: Path, batch_size: int = 500) -> int:
    """Insert trials into SQLite, skipping any with duplicate nct_id.

    Args:
        trials: List of Trial objects to persist.
        db_path: Path to the SQLite database file.
        batch_size: Number of rows per INSERT batch.

    Returns:
        Number of trials actually inserted (duplicates are skipped).
    """
    init_db(db_path)
    engine = get_engine(db_path)

    # Fetch existing NCT IDs to skip duplicates without relying on DB exceptions
    with Session(engine) as session:
        existing = {r[0] for r in session.execute(text("SELECT nct_id FROM trials"))}

    new_trials = [t for t in trials if t.nct_id not in existing]
    logger.info("Inserting %d new trials (%d already in DB)", len(new_trials), len(existing))

    inserted = 0
    with Session(engine) as session:
        for i in range(0, len(new_trials), batch_size):
            batch = [_to_row(t) for t in new_trials[i : i + batch_size]]
            session.add_all(batch)
            session.commit()
            inserted += len(batch)
            logger.info("Stored %d / %d trials...", inserted, len(new_trials))

    logger.info("Done. %d trials stored.", inserted)
    return inserted


def load_trials(db_path: Path) -> list[Trial]:
    """Load all trials from SQLite.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        List of all Trial objects.
    """
    engine = get_engine(db_path)
    with Session(engine) as session:
        rows = session.query(TrialRow).all()
    return [_from_row(row) for row in rows]
