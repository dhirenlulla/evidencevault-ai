from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    """
    Base class inherited by every SQLAlchemy ORM model.

    SQLAlchemy collects table metadata from all model classes that inherit
    from this base. Alembic later reads this metadata to generate migrations.
    """

    pass