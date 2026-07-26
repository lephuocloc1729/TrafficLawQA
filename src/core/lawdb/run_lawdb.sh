# Step 1. preprocess lawdb
python -m src.core.lawdb.preprocess_lawdb_infos
# Step 2. extract signs lawdb
python -m src.core.lawdb.extract_lawdb_sign_images
# Step 3. parse signs information (including sign name, description) lawdb
python -m src.core.lawdb.parse_lawdb_sign_infos
# Step 4. ingest signs information into qdrant
python -m src.core.lawdb.ingest_lawdb_signs
