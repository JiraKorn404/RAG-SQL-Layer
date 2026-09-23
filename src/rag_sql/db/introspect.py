"""Read table/column/key/comment metadata from the database (admin engine, used by indexing)."""

from dataclasses import dataclass, field

from sqlalchemy import Engine, inspect

# Tables owned by the tooling, never described to the model.
EXCLUDED_TABLE_PREFIXES = ("langchain_pg_",)


@dataclass(frozen=True)
class ColumnDoc:
    name: str
    type: str
    nullable: bool
    comment: str | None = None


@dataclass(frozen=True)
class ForeignKeyDoc:
    columns: list[str]
    referred_table: str
    referred_columns: list[str]


@dataclass(frozen=True)
class TableDoc:
    schema: str
    name: str
    comment: str | None
    columns: list[ColumnDoc]
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKeyDoc] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        return self.name if self.schema == "public" else f"{self.schema}.{self.name}"

    def to_text(self) -> str:
        """Compact, LLM-friendly description of the table."""
        lines = [f"Table {self.qualified_name}"]
        if self.comment:
            lines.append(f"  -- {self.comment}")
        for col in self.columns:
            flags = []
            if col.name in self.primary_key:
                flags.append("PRIMARY KEY")
            if not col.nullable and col.name not in self.primary_key:
                flags.append("NOT NULL")
            line = f"  {col.name} {col.type}"
            if flags:
                line += " " + " ".join(flags)
            if col.comment:
                line += f"  -- {col.comment}"
            lines.append(line)
        for fk in self.foreign_keys:
            lines.append(
                f"  FOREIGN KEY ({', '.join(fk.columns)}) "
                f"REFERENCES {fk.referred_table} ({', '.join(fk.referred_columns)})"
            )
        return "\n".join(lines)


def introspect_tables(engine: Engine, schemas: tuple[str, ...] = ("public",)) -> list[TableDoc]:
    insp = inspect(engine)
    tables: list[TableDoc] = []
    for schema in schemas:
        for name in sorted(insp.get_table_names(schema=schema)):
            if name.startswith(EXCLUDED_TABLE_PREFIXES):
                continue
            columns = [
                ColumnDoc(
                    name=c["name"],
                    type=str(c["type"]),
                    nullable=bool(c.get("nullable", True)),
                    comment=c.get("comment"),
                )
                for c in insp.get_columns(name, schema=schema)
            ]
            pk = insp.get_pk_constraint(name, schema=schema).get("constrained_columns") or []
            fks = [
                ForeignKeyDoc(
                    columns=fk["constrained_columns"],
                    referred_table=(
                        fk["referred_table"]
                        if fk.get("referred_schema") in (None, "public")
                        else f"{fk['referred_schema']}.{fk['referred_table']}"
                    ),
                    referred_columns=fk["referred_columns"],
                )
                for fk in insp.get_foreign_keys(name, schema=schema)
            ]
            tables.append(
                TableDoc(
                    schema=schema,
                    name=name,
                    comment=insp.get_table_comment(name, schema=schema).get("text"),
                    columns=columns,
                    primary_key=list(pk),
                    foreign_keys=fks,
                )
            )
    return tables
