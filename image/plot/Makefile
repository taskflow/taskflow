FILE_NAME=plot

read: pdf
	evince $(FILE_NAME).pdf &

pdf: ps
	ps2pdf $(FILE_NAME).ps

ps: dvi
	dvips -t letter -Pdownload35 -o $(FILE_NAME).ps $(FILE_NAME).dvi

dvi: 
	latex $(FILE_NAME).tex

clean: 
	rm -f $(FILE_NAME).{ps,pdf,log,aux,dvi,bbl}

backup: 
	tar -cvzf backup.tar.gz *

