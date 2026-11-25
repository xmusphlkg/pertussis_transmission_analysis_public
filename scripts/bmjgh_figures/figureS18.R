source("scripts/bmjgh_figures/supplement_style.R")

d <- read.csv("source_data/figures/figureS18.csv")
for (field in c("period_start", "period_end", "plot_date")) d[[field]] <- as.Date(d[[field]])
stopifnot(nrow(d) == 1186, length(unique(d$country)) == 8)
figure_country_labels <- setNames(country_levels,figure_countries)
plots <- list()
for (i in seq_along(figure_countries)) {
  country <- figure_countries[i]
  z <- d[d$country == country,]
  context <- unique(z[c("native_frequency","train_start","train_end")])
  zones <- do.call(rbind,lapply(unique(z$role),function(role) {
    x <- z[z$role==role,]
    data.frame(start=min(x$period_start),end=max(x$period_end),role=role)
  }))
  zones$window <- factor(zones$role,levels=c('training','posttraining_holdout','pretraining_backcast'),
    labels=c('Fitting','Post-training','Pretraining backcast'))
  identity <- paste(figure_country_labels[[country]],switch(context$native_frequency,
     monthly='monthly',weekly='weekly',mixed_annual_quarterly_or_other='annual / quarterly'),sep=' | ')
  z$setting <- identity
  end_date <- max(z$period_end)
  cutoff <- as.Date(paste0(context$train_end+1,'-01-01'))
  start_date <- min(z$period_start)
  breaks <- seq(as.Date(paste0(context$train_start,'-01-01')),end_date,by='4 years')
  breaks <- unique(c(as.Date(paste0(format(start_date,'%Y'),'-01-01')),breaks,
                     as.Date(paste0(format(end_date-1,'%Y'),'-01-01'))))
  if(length(breaks)>4) breaks <- breaks[c(1,ceiling(length(breaks)/2),length(breaks))]
  p <- ggplot(z)+
    geom_rect(data=zones,aes(xmin=start,xmax=end,ymin=-Inf,ymax=Inf,fill=window),inherit.aes=FALSE,colour=NA,
              show.legend=c(fill=TRUE,colour=FALSE))+
    geom_segment(aes(x=period_start,xend=period_end,y=expected_cases,yend=expected_cases,colour='Predicted'),linewidth=.45)+
    geom_point(aes(x=plot_date,y=observed_cases,colour='Observed'),size=.65)+
    scale_colour_manual(values=c('Observed'='#222222','Predicted'='#0072B2'),name=NULL,breaks=c('Observed','Predicted'))+
    scale_fill_manual(values=c('Fitting'='white','Post-training'='#DCEBF4','Pretraining backcast'='#EEEEEE'),
      limits=c('Fitting','Post-training','Pretraining backcast'),drop=FALSE,name=NULL)+
    scale_x_date(breaks=breaks,date_labels='%Y',expand=expansion(mult=c(.015,.025)))+
    scale_y_continuous(labels=scales::label_number(scale_cut=scales::cut_short_scale()),
      expand=expansion(mult=c(.02,.22)))+
    facet_wrap(~setting,ncol=1)+
    labs(x='Calendar year',y='Cases per reporting interval',tag=letters[i])+
    theme(plot.title=element_blank(),plot.subtitle=element_blank(),strip.text=element_text(size=7),
          legend.key.width=grid::unit(4,'mm'),legend.margin=margin(0,0,0,0),legend.box='vertical')+
    guides(colour=guide_legend(order=1,override.aes=list(shape=c(16,NA),linetype=c(0,1),linewidth=c(0,.45))),
           fill=guide_legend(order=2,override.aes=list(colour='#999999',linewidth=.2)))
  if(country %in% c('Australia','England'))
    p <- p+annotate('text',x=start_date+(end_date-start_date)*.04,y=max(c(z$observed_cases,z$expected_cases))*1.15,
                    label='No post-training observations',hjust=0,size=7/.pt)
  plots[[i]] <- p
}
composite <- wrap_plots(plots,ncol=2,guides='collect') & theme(legend.position='bottom',legend.box='vertical')

export_supplement(composite, "S18", 183, 260)
