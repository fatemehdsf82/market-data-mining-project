import os
import sys
import django

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'market.settings')
django.setup()

from django.db import connection

def check_table_schema(table_name):
    with connection.cursor() as cursor:
        # Get column information for SQL Server
        cursor.execute("""
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                CHARACTER_MAXIMUM_LENGTH,
                COLUMNPROPERTY(OBJECT_ID(TABLE_SCHEMA+'.'+TABLE_NAME), COLUMN_NAME, 'IsIdentity') as IS_IDENTITY
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION
        """, [table_name])
        
        columns = cursor.fetchall()
        print(f"\n=== Table: {table_name} ===")
        for col in columns:
            col_name, data_type, nullable, max_length, is_identity = col
            identity_info = " (IDENTITY)" if is_identity else ""
            length_info = f"({max_length})" if max_length else ""
            nullable_info = "NULL" if nullable == 'YES' else "NOT NULL"
            print(f"{col_name}: {data_type}{length_info} {nullable_info}{identity_info}")

# Check all main tables
tables = ['transactions', 'product', 'household', 'campaign', 'coupon', 'coupon_redemption', 'campaign_member']

for table in tables:
    try:
        check_table_schema(table)
    except Exception as e:
        print(f"Error checking {table}: {e}")

print("\nDone!")