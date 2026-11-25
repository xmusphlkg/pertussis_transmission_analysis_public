source("scripts/bmjgh_figures/supplement_style.R")

mechanism_colours <- c('Onward contribution'='#D55E00','Equal allocation'='#747678','Acquisition'='#0072B2')
z <- read.csv("source_data/figures/figureS22.csv");z <- z[z$horizon==30,];z$mechanism <- factor(z$allocation,levels=c(0,.5,1),labels=c('Onward contribution','Equal allocation','Acquisition'));z$setting <- factor(gsub('_',' ',z$country),levels=gsub('_',' ',figure_countries))
p <- ggplot(z,aes(duration,dalys,colour=mechanism))+geom_hline(yintercept=0,colour='#BBBBBB',linewidth=.25)+geom_line(linewidth=.4)+geom_point(size=1.4)+facet_wrap(~setting,ncol=2,scales='free_y')+scale_colour_manual(values=mechanism_colours,name=NULL)+scale_x_continuous(breaks=c(1,3,6,10))+labs(x='Mean protection duration (years)',y='30-year DALYs averted')+theme(legend.position='bottom')

export_supplement(p, "S22", 183, 205)
