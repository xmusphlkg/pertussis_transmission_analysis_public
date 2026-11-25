source("scripts/bmjgh_figures/supplement_style.R")

d <- read.csv("source_data/figures/figureS5.csv")
keep <- c('adopted_unchanged','recovery_365_unchanged','adopted_more_waned','adopted_more_recovered')
d <- d[d$scenario %in% keep & ((d$metric=='dalys' & d$age_group=='all_ages') | (d$metric=='deaths' & d$age_group=='infant_0')),]
stopifnot(nrow(d)==800,!anyDuplicated(d[c('country','scenario','year','metric')]))
d <- d[order(d$country,d$scenario,d$metric,d$year),]
d$cumulative <- ave(d$averted,interaction(d$country,d$scenario,d$metric),FUN=cumsum)
labels <- c(adopted_unchanged='Adopted epidemic history',recovery_365_unchanged='365-day contact recovery',adopted_more_waned='10% recovered to waned',adopted_more_recovered='10% waned to recovered')
colours <- c(adopted_unchanged='#0072B2',recovery_365_unchanged='#D55E00',adopted_more_waned='#777777',adopted_more_recovered='#777777')
lines <- c(adopted_unchanged='solid',recovery_365_unchanged='solid',adopted_more_waned='dashed',adopted_more_recovered='dotted')
d$scenario <- factor(d$scenario,levels=keep)
panels <- list()
for(metric in c('dalys','deaths')) for(kind in c('annual','cumulative')) for(country in c('Brazil','Japan')) {
 z <- d[d$country==country & d$metric==metric,]
 z$value <- if(kind=='annual') z$averted else z$cumulative
 ylabel <- paste(if(kind=='annual') 'Annual' else 'Cumulative',if(metric=='dalys') 'DALYs averted' else 'infant deaths\naverted')
 p <- ggplot(z,aes(year,value,colour=scenario,linetype=scenario))+
   geom_hline(yintercept=0,colour='#AAAAAA',linewidth=.2)+geom_line(linewidth=.4)+
   scale_colour_manual(values=colours,labels=labels,drop=FALSE)+scale_linetype_manual(values=lines,labels=labels,drop=FALSE)+
   scale_x_continuous(breaks=c(2027,2047,2067,2076),expand=expansion(mult=c(.02,.03)))+
   labs(x=if(metric=='deaths' && kind=='cumulative') 'Year' else NULL,y=ylabel,title=NULL,colour=NULL,linetype=NULL)+
   guides(colour=guide_legend(nrow=2,byrow=TRUE),linetype=guide_legend(nrow=2,byrow=TRUE))
 panels[[length(panels)+1]] <- p + facet_wrap(~country)
}
figure <- wrap_plots(panels,ncol=2,guides='collect')+plot_annotation(tag_levels='a') & theme(legend.position='bottom')

export_supplement(figure, "S5", 183, 200)
