#!/bin/bash
FN_JSON=$1
FN_CUTELOG_JSON="${FN_JSON%.log}_cutelog.log"
sed -e 'H;${x;s/\n/, \n/g;s/^,//;p;};d' "${FN_JSON}" | sed -e '1s/^/[/' |sed -e "\$a]" > "${FN_CUTELOG_JSON}"
