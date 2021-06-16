#!/usr/bin/env bash

python3 -m pip install --extra-index-url https://test.pypi.org/simple nvflare==0.9.7
# python3 -m pip install nvflare*.whl
provision -n
mkdir -p exp

unzip -o packages/server.zip -d exp/server
unzip -o packages/site-1.zip -d exp/site-1
unzip -o packages/site-2.zip -d exp/site-2
unzip -o "packages/admin@nvidia.com.zip" -d exp/admin

cp -rf transfer exp/admin/

python3 -m pip install torch torchvision

echo "-------> Starting server"
bash exp/server/startup/start.sh
sleep 20
echo "-------> Starting client site-1"
bash exp/site-1/startup/start.sh
sleep 10
echo "-------> Starting client site-2"
bash exp/site-2/startup/start.sh
sleep 10

./interactive_admin
# exp/admin/startup/fl_admin.sh < admin_input.txt


