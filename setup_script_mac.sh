brew install protobuf
if [ -f "${HOME}/.zshrc" ]; then
    echo 'export PATH="/opt/homebrew/opt/protobuf/bin:$PATH"' >> ~/.zshrc
elif [ -f "${HOME}/.bashrc" ]; then
    echo 'export PATH="/opt/homebrew/opt/protobuf/bin:$PATH"' >> ~/.bashrc
else
    echo "No .zshrc or .bashrc file found in home directory."
    echo "Please create one of these files and add the following line:"
    echo "export PATH=/opt/homebrew/opt/protobuf/bin:\$PATH"
    echo "Then run the script again."
    exit 1
fi
# Harder solution that does not work
# cwd=$(pwd)
# git clone --recursive https://github.com/grpc/grpc
# cd grpc
# make plugins -j 12

# NOTE: Might be an easier solution here: https://github.com/grpc/grpc/issues/15675
python3 -m venv venv
source venv/bin/activate
pip install grpclib protobuf grpcio grpcio-tools tabulate==0.9.0 hydra-core
