source("scripts/bmjgh_figures/supplement_style.R")

countries <- figure_countries
country_names <- setNames(country_levels, countries)
start_colours <- c('2027'='#084594','2029'='#2171B5','2032'='#6BAED6')
trajectory <- read.csv("source_data/figures/figureS7.csv")
trajectory$country <- factor(trajectory$country, levels=countries)
trajectory$start_year <- factor(trajectory$start_year, levels=c(2027,2029,2032))
trajectory$population <- factor(trajectory$population, levels=c("All ages","Adolescents (10–19 years)"))
nochange <- unique(trajectory[trajectory$strategy_id == "current","country",drop=FALSE])
nochange$label_x <- 10
nochange$label_y <- Inf
nochange$label_text <- "No coverage increase"
f3<-ggplot(trajectory,aes(x=followup_year,y=cumulative_cases_averted,colour=start_year,linetype=population,
                         group=interaction(start_year,population)))+
  geom_hline(yintercept=0,colour='#B0B0B0',linewidth=.25)+geom_line(linewidth=.45)+
  geom_text(data=nochange,aes(x=label_x,y=label_y,label=label_text),inherit.aes=FALSE,
            colour='#555555',family='Arial',size=7/.pt,vjust=1.8)+
  facet_wrap(~country,ncol=2,scales='free_y',labeller=as_labeller(country_names))+
  scale_colour_manual(values=start_colours,name='Start year')+
  scale_linetype_manual(values=c('All ages'='solid','Adolescents (10–19 years)'='22'),name=NULL)+
  scale_x_continuous(breaks=c(0,5,10,15,20),limits=c(0,20),expand=expansion(mult=c(.015,.025)))+
  scale_y_continuous(labels=scales::label_number(scale_cut=scales::cut_short_scale()),
                    breaks=function(limits) if(max(abs(limits))<1) 0 else scales::breaks_pretty(n=4)(limits),
                    expand=expansion(mult=c(.08,.1)))+
  labs(title='Reported-case effects across programme start years',
       subtitle='Adolescent coverage expanded towards 70%; positive values indicate fewer cases',
       x='Years since programme start',y='Cumulative reported cases averted')+
  theme(legend.position='bottom',legend.box='vertical',legend.margin=margin(0,0,0,0),
        panel.spacing=grid::unit(3,'mm'),strip.text=element_text(size=8),
        legend.key.width=grid::unit(8,'mm'))+
  guides(colour=guide_legend(order=1,nrow=1),linetype=guide_legend(order=2,nrow=1))
f3 <- f3 + labs(title=NULL, subtitle=NULL)

export_supplement(f3, "S7", 183, 176)
