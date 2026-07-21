#!/bin/bash
# WSL 网络诊断: 找能连通 pip 的路径 (proxy / mirror)。
echo "=== gateway ==="
GW=$(ip route show default | awk '{print $3}')
echo "GW=$GW"

echo "=== proxy ports on GW ==="
for p in 7890 7897 10809 10808 1080 2080 8888 7891 20171; do
  if timeout 2 bash -c "echo > /dev/tcp/$GW/$p" 2>/dev/null; then
    echo "OPEN: $p"
  fi
done
echo "(end ports)"

echo "=== direct mirror (no proxy) ==="
for m in https://pypi.org/simple/pip/ https://mirrors.aliyun.com/pypi/simple/pip/ https://pypi.mirrors.ustc.edu.cn/simple/pip/ https://download.pytorch.org/whl/cu126/torch/; do
  code=$(timeout 10 curl -s -o /dev/null -w "%{http_code}" "$m" 2>/dev/null || echo "TIMEOUT")
  echo "$code  $m"
done

echo "=== via proxy (try each open port) ==="
for p in 7890 7897 10809 10808 1080; do
  code=$(timeout 10 curl -s -o /dev/null -w "%{http_code}" -x "http://$GW:$p" https://pypi.org/simple/pip/ 2>/dev/null || echo "FAIL")
  echo "$code  pypi.org via $GW:$p"
done
