all: html

build/%.ipynb: %.md environment.yml utils.py
	@mkdir -p $(@D)
	cd $(@D); python ../md2ipynb.py ../../$< ../../$@

build/%.md: %.md
	@mkdir -p $(@D)
	@cp $< $@

MARKDOWN = $(wildcard chapter_preface/*.md */index.md)
NOTEBOOK = $(filter-out $(MARKDOWN), $(wildcard chapter*/*.md))

OBJ = $(patsubst %.md, build/%.md, $(MARKDOWN)) \
	$(patsubst %.md, build/%.ipynb, $(NOTEBOOK))

DEPS = build/img build/data build/environment.yml build/utils.py build/README.md

PKG = build/_build/html/gluon_tutorials.tar.gz build/_build/html/gluon_tutorials.zip

pkg: $(PKG)

build/_build/html/gluon_tutorials.zip: $(OBJ) $(DEPS)
	cd build; zip -r $(patsubst build/%, %, $@ $(DEPS)) chapter*

build/_build/html/gluon_tutorials.tar.gz: $(OBJ) $(DEPS)
	cd build; tar -zcvf $(patsubst build/%, %, $@ $(DEPS)) chapter*

build/%: %
	@cp $< $@

build/img:
	rsync -rupE img build/

build/data:
	rsync -rupE data build/

html: $(DEPS) $(OBJ)
	make -C build html

latex: $(DEPS) $(OBJ)
	make -C build latex
	cd build/_build/latex; make

clean:
	rm -rf build/chapter* $(DEPS) $(PKG)
