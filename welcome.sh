#!/usr/bin/env bash

# python3 -m pip install -i https://test.pypi.org/simple/ nvflare
# provision -n
# mkdir exp
# unzip -fo packages/server.zip -d exp/server
# unzip -fo packages/site-1.zip -d exp/site-1
# unzip -fo packages/site-2.zip -d exp/site-2
# unzip -fo "packages/admin@nvidia.com.zip" -d exp/admin
# 
bash exp/server/startup/start.sh
sleep 10
bash exp/site-1/startup/start.sh
bash exp/site-2/startup/start.sh
sleep 10

echo "Now enter admin@nvidia.com for login"
bash exp/admin/startup/fl_admin.sh 

