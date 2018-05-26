set @dtref = '2018-04-01';
set @dtmoins1 = @dtref - interval 1 year;

delete from tmp.scores_predictions;

insert into tmp.scores_predictions (SIREN) (select SIREN from tmp.scores_stats);
#where siren<672050085

# Table insee.identite
# naf = ape_ent
update tmp.scores_predictions tt inner join insee.identite ii on ii.siren=tt.siren and ii.siege=1 and ii.actif=1
set
tt.ii_ACTIVNAT=ii.ACTIVNAT,tt.ii_ORIGINE=ii.ORIGINE,tt.ii_MODET=ii_MODET, tt.ii_EXPLET=ii.EXPLET, tt.ii_DAPET=ii.DAPET, 
tt.ii_CJ=ii.CJ, tt.ii_NBETAB=ii.NBETAB, tt.ii_APE_ENT=ii.APE_ENT, tt.ii_PROCOL=ii.PROCOL, tt.ii_CAPITAL=ii.CAPITAL, 
tt.ii_EFF_ENT=ii.EFF_ENT, tt.ii_TEFF_ENT=ii.TEFF_ENT, 
tt.ii_ADR_DEP=ii.ADR_DEP, tt.ii_TCA=ii.TCA, tt.ii_TCAEXP=ii_TCAEXP;

# Tables jo.lienRef et jo.liens2
update tmp.scores_predictions tt set jl_PARTICIPATION=
(select count(*) from jo.liens2 where idPar in (select id from jo.liensRef as jl where jl.siren=tt.siren) and actif=1 and PDetention>=33);
update tmp.scores_predictions tt set jl_ACTIONNARIAT=
(select count(*) from jo.liens2 where idAct in (select id from jo.liensRef as jl where jl.siren=tt.siren) and actif=1 and PDetention>=33);

# Table jo.dirigeants
update tmp.scores_predictions tt set jd_NBPM=
(select count(*) from jo.dirigeants as jd where typeDir='PM' and jd.dirsiren=tt.siren);
update tmp.scores_predictions tt set jd_NBPP=
(select count(*) from jo.dirigeants as jd where typeDir='PP' and jd.dirsiren=tt.siren);

# Table sdv1.bourse_isin
update tmp.scores_predictions tt set sb_EnBourse=
(select 1 from sdv1.bourse_isin as sb where sb.siren=tt.siren);



# TODO : intégrer le temps

# Table jo.greffes_affaires_siren
update tmp.scores_predictions tt set jg_NBDE=
(select count(*) from jo.greffes_affaires_siren as jg where jg.qualite='DE' and jg.entSiren=tt.siren and dateInsert<@dtmoins1);
update tmp.scores_predictions tt set jg_NBDF=
(select count(*) from jo.greffes_affaires_siren as jg where jg.qualite='DF' and jg.entSiren=tt.siren and dateInsert<@dtmoins1);

# Table bopi.marques
update tmp.scores_predictions tt set bm_NBMARQUES=
(select count(*) from bopi.marques as bm where bm.sirenDeposant=tt.siren);

# Table jo.bilans_postes
set @exoref = 20171231;
set @exoref1 = 20161231;
set @exoref2 = 20151231;
update tmp.scores_predictions tt inner join jo.bilans_postes jb on jb.siren=tt.siren 
and jb.dateExercice = @exoref 
and jb.liasse='2050' and jb.monnaie='EUR' and jb.dureeExercice=12 and (select typeBilan from jo.bilans bb where jb.id=bb.id)='N' 
set jb_BK=jb.BK,jb_BK1=jb.BK1,jb_CK=jb.CK,jb_CK1=jb.CK1,jb_DI=jb.DI,jb_DL=jb.DL,jb_DO=jb.DO,jb_DR=jb.DR,
jb_EC=jb.EC,jb_EE=jb.EE,jb_FJ=jb.FJ,jb_FK=jb.FK,jb_FR=jb.FR,jb_GF=jb.GF,jb_GP=jb.GP,jb_GU=jb.GU,
jb_GV=jb.GV,jb_GW=jb.GW,jb_HD=jb.HD,jb_HH=jb.HH,jb_HN=jb.HN;

update tmp.scores_predictions tt inner join jo.bilans_postes jb on jb.siren=tt.siren inner join jo.bilans bb on jb.id=bb.id 
and jb.dateExercice = @exoref1 
and jb.liasse='2050' and jb.monnaie='EUR' and jb.dureeExercice=12 and bb.typeBilan='N' 
set jb_1_BK=jb.BK,jb_1_BK1=jb.BK1,jb_1_CK=jb.CK,jb_1_CK1=jb.CK1,jb_1_DI=jb.DI,jb_1_DL=jb.DL,jb_1_DO=jb.DO,
jb_1_DR=jb.DR,jb_1_EC=jb.EC,jb_1_EE=jb.EE,jb_1_FJ=jb.FJ,jb_1_FK=jb.FK,jb_1_FR=jb.FR,jb_1_GF=jb.GF,
jb_1_GP=jb.GP,jb_1_GU=jb.GU,jb_1_GV=jb.GV,jb_1_GW=jb.GW,jb_1_HD=jb.HD,jb_1_HH=jb.HH,jb_1_HN=jb.HN;

update tmp.scores_predictions tt inner join jo.bilans_postes jb on jb.siren=tt.siren inner join jo.bilans bb on jb.id=bb.id 
and jb.dateExercice = @exoref2 
and jb.liasse='2050' and jb.monnaie='EUR' and jb.dureeExercice=12 and bb.typeBilan='N' 
set jb_2_BK=jb.BK,jb_2_BK1=jb.BK1,jb_2_CK=jb.CK,jb_2_CK1=jb.CK1,jb_2_DI=jb.DI,jb_2_DL=jb.DL,jb_2_DO=jb.DO,
jb_2_DR=jb.DR,jb_2_EC=jb.EC,jb_2_EE=jb.EE,jb_2_FJ=jb.FJ,jb_2_FK=jb.FK,jb_2_FR=jb.FR,jb_2_GF=jb.GF,
jb_2_GP=jb.GP,jb_2_GU=jb.GU,jb_2_GV=jb.GV,jb_2_GW=jb.GW,jb_2_HD=jb.HD,jb_2_HH=jb.HH,jb_2_HN=jb.HN;



#TODO : web

select * from tmp.scores_predictions where substr(ii_CJ,1,1)=5;
