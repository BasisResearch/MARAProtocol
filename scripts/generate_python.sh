#!/bin/bash

# Create necessary directories if they don't exist
mkdir -p generated
mkdir -p generated/mara

# List of proto files to compile
PROTO_FILES=(
  "mara_environment.proto"
  "mara_environment_service.proto"
  "mara_registry.proto"
  "mara_agent.proto"
  "mara_evaluation.proto"
  "mara_evaluation_controller.proto"
)

# Compile each proto file
for proto_file in "${PROTO_FILES[@]}"; do
  echo "Compiling $proto_file..."
  
  python -m grpc_tools.protoc \
    --proto_path=./protocols \
    --python_out=./generated/mara \
    --grpc_python_out=./generated/mara \
    "./protocols/$proto_file"
  
  # Check if compilation was successful
  if [ $? -eq 0 ]; then
    echo "✅ Successfully compiled $proto_file"
  else
    echo "❌ Failed to compile $proto_file"
    exit 1
  fi
done

# Fix imports in generated files
echo "Fixing imports in generated files..."
# Fix "from mara_X import" style imports
find ./generated/mara -name "*.py" -type f -exec sed -i.bak 's/from mara_/from generated.mara.mara_/g' {} \;
find ./generated/mara -name "*.py.bak" -type f -delete

# Fix "import mara_X" style imports
find ./generated/mara -name "*.py" -type f -exec sed -i.bak 's/import mara_/import generated.mara.mara_/g' {} \;
find ./generated/mara -name "*.py.bak" -type f -delete

echo "All proto files compiled successfully!"
