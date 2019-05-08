# Author: zhanglongx <zhanglongx@gmail.com>

VERBOSE?=@

ALHENA2:=.
CACHE_H5_CN:=$(ALHENA2)/database/cn/cn_database.h5
CACHE_DAILY_CN:=$(wildcard $(ALHENA2)/database/cn/daily/*/*.csv)
CACHE_REPORT_CN:=$(wildcard $(ALHENA2)/database/cn/report/*/*.csv)
__DAILY_TAR:=database.tar.xz

.PHONY : help update h5 clean

help:
	$(VERBOSE) echo "Alhena2: "
	$(VERBOSE) echo "    make update            update cache file(not .h5 file),"
	$(VERBOSE) echo "                           can take up a *VERY* long time"
	$(VERBOSE) echo "    make h5                build .h5 file"
	$(VERBOSE) echo "    make database.tar.xz   database tar"
	$(VERBOSE) echo "    make clean             clean .h5 file"

update:
	$(VERBOSE) python3 ./reader.py update

build: $(CACHE_H5_CN)

$(CACHE_H5_CN): $(CACHE_DAILY_CN) $(CACHE_REPORT_CN)
	$(VERBOSE) python3 ./reader.py -f $@ build

$(__DAILY_TAR):
	$(VERBOSE) ssh Bandwagon2 "cd ~/Workdir/Alhena && ./utilities/tar_database.sh"
	$(VERBOSE) scp zhlx@Bandwagon2:/home/zhlx/Workdir/Alhena/database.tar.xz .

clean:
	$(VERBOSE) rm -f $(CACHE_H5_CN)