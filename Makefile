# Author: zhanglongx <zhanglongx@gmail.com>

VERBOSE?=@

ALHENA2:=.
CACHE_H5_CN:=$(ALHENA2)/all_cn.h5
CACHE_DAILY_CN:=$(wildcard $(ALHENA2)/database/cn/daily/*.csv)
CACHE_REPORT_CN:=$(wildcard $(ALHENA2)/database/cn/report/*.csv)

.PHONY : help update h5 clean

help:
	$(VERBOSE) echo "Alhena2: "
	$(VERBOSE) echo "    make update    update cache file(not .h5 file), can take up a"
	$(VERBOSE) echo "                   *VERY* long time"
	$(VERBOSE) echo "    make h5        build .h5 file"
	$(VERBOSE) echo "    make clean     clean .h5 file"

update:
	$(VERBOSE) python3 ./reader.py update

h5: $(CACHE_H5_CN)

$(CACHE_H5_CN): $(CACHE_DAILY_CN) $(CACHE_REPORT_CN)
	$(VERBOSE) python3 ./reader.py -s cn -f $@ build

clean:
	$(VERBOSE) rm -f $(CACHE_H5_CN)