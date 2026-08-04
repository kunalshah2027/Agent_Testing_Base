import os
from datetime import datetime
from typing import Any, Dict, List
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, insert
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

engine = create_engine("sqlite:///:memory:")
metadata = MetaData()

payments_table = Table(
    "payments",
    metadata,
    Column("payment_id", String(50), primary_key=True),
    Column("amount", Integer),
    Column("currency", String(3)),
    Column("status", String(20)),  # e.g., 'Completed', 'Repair'
    Column("error_code", String(20)),  # e.g., 'ERR_001', 'ERR_002'
    Column("error_description", Text)
)

metadata.create_all(engine)

# Insert mock enterprise payment data matching your QA use cases
with engine.connect() as connection:
    connection.execute(
        insert(payments_table),
        [
            {
                "payment_id": "TXN_1001",
                "amount": 5000,
                "currency": "USD",
                "status": "Repair",
                "error_code": "ERR_4041",
                "error_description": "Beneficiary BIC bank code not found in SWIFT directory."
            },
            {
                "payment_id": "TXN_1002",
                "amount": 12500,
                "currency": "EUR",
                "status": "Repair",
                "error_code": "ERR_2019",
                "error_description": "Insufficient funds in client clearing account for structural settlement."
            },
            {
                "payment_id": "TXN_1003",
                "amount": 750,
                "currency": "USD",
                "status": "Completed",
                "error_code": None,
                "error_description": None
            }
        ]
    )
    connection.commit()

# Wrap the SQLAlchemy engine into LangChain's SQLDatabase utility
db = SQLDatabase(engine)
# --- Verification Tests ---
print("\n--- 1. Testing LangChain SQLDatabase Wrapper ---")
# Check if LangChain can discover the table name
print("Discovered Tables:", db.get_usable_table_names())

# Check if LangChain can fetch the table schema structure
print("\nTable Schema Layout:")
print(db.get_table_info(["payments"]))

# Run a test query through LangChain's utility wrapper
print("\nQuerying Data via LangChain:")
test_query_result = db.run("SELECT * FROM payments WHERE status = 'Repair';")
print(test_query_result)
