source("scripts/bmjgh_figures/supplement_style.R")

ml <- read.csv("source_data/figures/figureS14.csv")
ml$component <- factor(ml$component, levels = c("Infant years of life lost","Years lived with disability"))
m <- ml[!duplicated(ml$maternal_coverage), ]
mm <- ggplot(ml,aes(factor(round(maternal_coverage*100)),value,fill=component))+
 geom_col(width=.56,position=position_stack(reverse=TRUE))+
 geom_errorbar(data=m,aes(x=factor(round(maternal_coverage*100)),ymin=dalys_averted_q025,ymax=dalys_averted_q975),
               inherit.aes=FALSE,width=.12,linewidth=.3)+
 scale_fill_manual(values=c('#0072B2','#E69F00'),name=NULL)+
 scale_y_continuous(expand=expansion(mult=c(0,.08)))+
 labs(x='Maternal coverage (%)',y='Additional DALYs averted by nasal vaccination',title='China: components of marginal nasal benefit')+
 theme(legend.position='bottom')
mm <- mm + labs(title=NULL) + facet_wrap(~country)

export_supplement(mm, "S14", 140, 100)
