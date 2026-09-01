# app/routes/pgvector_routes.py
from flask import Blueprint, jsonify

from app.async_runner import async_route
from app.errors import APIError
from app.services.database import PSQLDatabase
from app.validation import query_param, required_query_param, required_query_params

bp = Blueprint("pgvector", __name__)


async def check_index_exists(table_name: str, column_name: str) -> bool:
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetch(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_indexes
                WHERE tablename = $1 
                AND indexdef LIKE '%' || $2 || '%'
            );
            """,
            table_name,
            column_name,
        )
    return result[0]['exists']


@bp.get("/test/check_index")
@async_route
async def check_file_id_index():
    table_name, column_name = required_query_params("table_name", "column_name")
    if await check_index_exists(table_name, column_name):
        return jsonify({"message": f"Index on {column_name} exists in the table {table_name}."})
    else:
        # Always answered 200 with the error object as the body (debug route).
        return jsonify(
            {
                "status_code": 404,
                "detail": f"No index on {column_name} found in the table {table_name}.",
                "headers": None,
            }
        )


@bp.get("/db/tables")
@async_route
async def get_table_names():
    schema = query_param("schema", "public")
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        table_names = await conn.fetch(
            """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = $1
            """,
            schema,
        )
    # Extract table names from records
    tables = [record['table_name'] for record in table_names]
    return jsonify({"schema": schema, "tables": tables})


@bp.get("/db/tables/columns")
@async_route
async def get_table_columns():
    table_name = required_query_param("table_name")
    schema = query_param("schema", "public")
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        columns = await conn.fetch(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
            ORDER BY ordinal_position;
            """,
            schema, table_name,
        )
    column_names = [col['column_name'] for col in columns]
    return jsonify({"table_name": table_name, "columns": column_names})


@bp.get("/records/all")
@async_route
async def get_all_records():
    table_name = required_query_param("table_name")
    # Validate that the table name is one of the expected ones to prevent SQL injection
    if table_name not in ["langchain_pg_collection", "langchain_pg_embedding"]:
        raise APIError(status_code=400, detail="Invalid table name")

    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # Use SQLAlchemy core or raw SQL queries to fetch all records
        records = await conn.fetch(f"SELECT * FROM {table_name};")

    # Convert records to JSON serializable format, assuming records can be directly serialized
    records_json = [dict(record) for record in records]

    return jsonify(records_json)


@bp.get("/records")
@async_route
async def get_records_filtered_by_custom_id():
    custom_id = required_query_param("custom_id")
    table_name = query_param("table_name", "langchain_pg_embedding")
    # Validate that the table name is one of the expected ones to prevent SQL injection
    if table_name not in ["langchain_pg_collection", "langchain_pg_embedding"]:
        raise APIError(status_code=400, detail="Invalid table name")

    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # Use parameterized queries to prevent SQL Injection
        query = f"SELECT * FROM {table_name} WHERE custom_id=$1;"
        records = await conn.fetch(query, custom_id)

    # Convert records to JSON serializable format, assuming the Record class has a dict method.
    records_json = [dict(record) for record in records]

    return jsonify(records_json)
