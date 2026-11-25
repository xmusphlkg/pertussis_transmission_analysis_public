source("scripts/bmjgh_figures/supplement_style.R")

data <- read.csv("source_data/figures/figureS19.csv")
countries <- c("Sweden","New_Zealand")
first <- as.Date("2021-01-01")
end <- as.Date("2026-01-01")
data$plot_date <- as.Date(data$plot_date)
panels <- list()
for (country in countries) for (cutoff in c(2022,2023)) {
  panels[[length(panels)+1]] <- data[data$country==country & data$cutoff==cutoff,]
}
limits <- list()
for(country in countries) {
  d<-data[data$country==country,];tick<-pretty(c(0,max(c(d$observed_cases,d$expected_cases))*1.07),n=4)
  limits[[country]]<-max(tick)
}
colours <- c('Observed'='#303030','Model'='#0072B2')
linetypes <- c('Observed'='solid','Model'='dashed')
labels <- c('Sweden','Sweden','New Zealand','New Zealand')
plots <- list()
for(i in seq_along(panels)) {
  d<-panels[[i]];country<-d$country[1];cutoff<-d$cutoff[1]
  edge<-as.Date(paste0(cutoff+1L,'-01-01'));shade<-data.frame(start=edge,end=end)
  d$setting <- paste0(labels[i], " · through ", cutoff)
  p<-ggplot(d,aes(x=plot_date))+facet_wrap(~setting)+
    geom_rect(data=shade,aes(xmin=start,xmax=end,ymin=-Inf,ymax=Inf,fill='After cutoff'),
              inherit.aes=FALSE,colour=NA,show.legend=c(fill=TRUE,colour=FALSE,linetype=FALSE))+
    geom_vline(xintercept=edge,colour='#757575',linewidth=.25,linetype='dotted')+
    geom_line(aes(y=observed_cases,colour='Observed',linetype='Observed'),linewidth=.35,
              show.legend=c(colour=TRUE,linetype=FALSE,fill=FALSE))+
    geom_point(aes(y=observed_cases,colour='Observed'),shape=21,fill='white',size=.85,stroke=.25,show.legend=FALSE)+
    geom_line(aes(y=expected_cases,colour='Model',linetype='Model'),linewidth=.50,
              show.legend=c(colour=TRUE,linetype=FALSE,fill=FALSE))+
    scale_colour_manual(name='Series',values=colours,breaks=c('Observed','Model'))+
    scale_linetype_manual(name='Series',values=linetypes,breaks=c('Observed','Model'))+
    scale_fill_manual(values=c('After cutoff'='#EDEFF0'))+
    guides(linetype='none',colour=guide_legend(override.aes=list(linetype=c('solid','dashed'))))+
    scale_x_date(breaks=seq(first,as.Date('2025-01-01'),by='year'),date_labels='%Y',
                 limits=c(first,end),expand=expansion(mult=0))+
    scale_y_continuous(limits=c(0,limits[[country]]),breaks=pretty(c(0,limits[[country]]),n=4),expand=expansion(mult=0))+
    labs(tag=letters[i],
         x=if(i>2) 'Calendar year' else NULL,
         y=if(i%%2L==1L) 'Reported cases per month' else NULL)+
    theme(legend.position='top',legend.box='horizontal',legend.key.width=grid::unit(9,'mm'),
          axis.text.x=if(i<=2) element_blank() else element_text(size=6,colour='#222222'),
          axis.text.y=if(i%%2L==0L) element_blank() else element_text(size=6,colour='#222222'),
          axis.ticks.y=if(i%%2L==0L) element_blank() else element_line(linewidth=.25,colour='#555555'))
  plots[[i]]<-p
}
figure <- (plots[[1]]|plots[[2]])/(plots[[3]]|plots[[4]])+
  plot_layout(guides='collect') & theme(legend.position='top')

export_supplement(figure, "S19", 183, 133)
