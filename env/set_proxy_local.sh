#!/bin/bash

# 设置HTTP代理（标准大写变量名）
export HTTP_PROXY="http://172.17.0.2:7532"
export HTTPS_PROXY="http://172.17.0.2:7532"

# 兼容小写变量名（可选）
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"

# 设置不代理的地址（根据你的网络环境调整）
export NO_PROXY="localhost,127.0.0.1,172.17.0.0/16"
export no_proxy="$NO_PROXY"

echo "HTTP/HTTPS proxies configured:"
echo " - HTTP_PROXY:  $HTTP_PROXY"
echo " - HTTPS_PROXY: $HTTPS_PROXY"
echo " - NO_PROXY:    $NO_PROXY"
