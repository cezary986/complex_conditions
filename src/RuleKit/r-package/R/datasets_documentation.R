#' @details Attribute information.
#' \itemize{
#'  \item{seismic} {result of shift seismic hazard assessment in the mine working obtained by the seismic
#'  method (a - lack of hazard, b - low hazard, c - high hazard, d - danger state)}
#'  \item{seismoacoustic}{result of shift seismic hazard assessment in the mine working obtained by the
#'  seismoacoustic method}
#'  \item{shift}{information about type of a shift (W - coal-getting, N -preparation shift)}
#'  \item{genergy}{seismic energy recorded within previous shift by the most active geophone (GMax) out of
#'  geophones monitoring the longwall}
#'  \item{gpuls}{a number of pulses recorded within previous shift by GMax}
#'  \item{gdenergy}{ a deviation of energy recorded within previous shift by GMax from average energy recorded
#'  during eight previous shifts}
#'  \item{gdpuls}{a deviation of a number of pulses recorded within previous shift by GMax from average number
#'  of pulses recorded during eight previous shifts}
#'  \item{ghazard}{result of shift seismic hazard assessment in the mine working obtained by the
#'  seismoacoustic method based on registration coming form GMax only}
#'  \item{nbumps}{the number of seismic bumps recorded within previous shift}
#'  \item{nbumps2}{the number of seismic bumps (in energy range [0^2,10^3)) registered within previous shift}
#'  \item{nbumps3}{the number of seismic bumps (in energy range [10^3,10^4)) registered within previous shift}
#'  \item{nbumps4}{the number of seismic bumps (in energy range [10^4,10^5)) registered within previous shift}
#'  \item{nbumps5}{the number of seismic bumps (in energy range [10^5,10^6)) registered within previous shift}
#'  \item{nbumps6}{the number of seismic bumps (in energy range [10^6,10^7)) registered within previous shift}
#'  \item{nbumps7}{the number of seismic bumps (in energy range [10^7,10^8)) registered within previous shift}
#'  \item{nbumps89}{the number of seismic bumps (in energy range [10^8,10^10)) registered within previous shift}
#'  \item{energy}{total energy of seismic bumps registered within previous shift}
#'  \item{maxenergy}{the maximum energy of the seismic bumps registered within previous shift}
#'  \item{class}{the decision attribute - "1" means that high energy seismic bump occurred in the next shift
#'  ("hazardous state"), "0" means that no high energy seismic bumps occurred in the next shift
#'  ("non-hazardous state").}}
#' @name seismic_bumps
#' @docType data
#' @title Seismoacoustic data form coal mine
#' @description The data describes the problem of high energy (higher than 10^4 J) seismic bumps
#' forecasting in a coal mine. Data comes from two of longwalls located in a Polish coal mine
#' @format A data frame with 2584 rows and 19 variables
#' @keywords datasets
NULL

#' @details Attribute information.
#' \itemize{
#'  \item{Recipientgender}{Male - 1, Female - 0}
#'  \item{Stemcellsource}{Source of hematopoietic stem cells (Peripheral blood - 1, Bone marrow - 0)}
#'  \item{Donorage}{Age of the donor at the time of hematopoietic stem cells apheresis}
#'  \item{Donorage35}{Donor age <35 - 0, Donor age >=35 - 1}
#'  \item{IIIV}{Development of acute graft versus host disease stage II or III or IV (Yes - 1, No - 0)}
#'  \item{Gendermatch}{Compatibility of the donor and recipient according to their gender (Female to Male - 1, Other - 0)}
#'  \item{DonorABO}{ABO blood group of the donor of hematopoietic stem cells (0 - 0, 1, A, B=-1, AB=2)}
#'  \item{RecipientABO}{ABO blood group of the recipient of hematopoietic stem cells (0 - 0, 1, A, B=-1, AB=2)}
#'  \item{RecipientRh}{Presence of the Rh factor on recipient???s red blood cells ('+' - 1, '-' - 0)}
#'  \item{ABOMatch}{Compatibility of the donor and the recipient of hematopoietic stem cells according to ABO blood group (matched - 1, mismatched - 1)}
#'  \item{CMVstatus}{Serological compatibility of the donor and the recipient of hematopoietic stem cells according to cytomegalovirus
#'  infection prior to transplantation (the higher the value the lower the compatibility)}
#'  \item{RecipientCMV}{Presence of cytomegalovirus infection in the donor of hematopoietic stem cells prior to transplantation (presence - 1, absence - 0)}
#'  \item{Disease}{Type of disease (ALL,AML,chronic,nonmalignant,lymphoma)}
#'  \item{Riskgroup}{High risk - 1, Low risk - 0}
#'  \item{Txpostrelapse}{The second bone marrow transplantation after relapse (No - 0; Yes - 1)}
#'  \item{Diseasegroup}{Type of disease (malignant - 1, nonmalignant - 0)}
#'  \item{HLAmatch}{ Compatibility of antigens of the main histocompatibility complex of the donor and the recipient of hematopoietic stem cells
#'  according to ALL international BFM SCT 2008 criteria (10/10 - 0, 9/10 - 1, 8/10 - 2, 7/10 - 3 (allele/antigens))}
#'  \item{HLAmismatch}{HLA matched - 0, HL mismatched - 1}
#'  \item{Antigen}{In how many anigens there is difference beetwen the donor nad the recipient (-1 - no differences, 0 - one difference,1 (2) - two (three) diffences)}
#'  \item{Allel}{In how many allele there is difference beetwen the donor nad the recipient [-1 no differences,0 - one difference, 1 (2) (3) - two, (tree, four) differences)}
#'  \item{HLAgrI}{The differecne type beetwien the donor and the recipient (HLA mateched - 0,the difference is in only one antigen - 1,
#'  the difference is only in one allel - 2, the difference is only in DRB1 cell - 3, two differences (two allele or two antignes) - 4,
#'  two differences (two allele or two antignes) - 5)}
#'  \item{Recipientage}{Age of the recipient of hematopoietic stem cells at the time of transplantation}
#'  \item{Recipientage10}{Recipient age <10 - 0, Recipient age>=10 - 1}
#'  \item{Recipientageint}{Recipient age in (0,5] - 0, (5, 10] - 1, (10, 20] - 2}
#'  \item{Relapse}{Reoccurrence of the disease (No - 0, Yes - 1)}
#'  \item{aGvHDIIIIV}{Development of acute graft versus host disease stage III or IV (Yes - 0, No - 1)}
#'  \item{extcGvHD}{Development of extensive chronic graft versus host disease (Yes - 0, No - 1)}
#'  \item{CD34kgx10d6}{CD34+ cell dose per kg of recipient body weight (10^6/kg)}
#'  \item{CD3dCD34}{CD3+ cell to CD34+ cell ratio}
#'  \item{CD3dkgx10d8}{CD3+ cell dose per kg of recipient body weight (10^8/kg)}
#'  \item{Rbodymass}{Body mass of the recipient of hematopoietic stem cells at the time of transplantation}
#'  \item{ANCrecovery}{Time to neutrophils recovery defined as neutrophils count >0.5 x 10^9/L}
#'  \item{PLTrecovery}{Time to platelet recovery defined as platelet count >50000/mm3}
#'  \item{time_to_aGvHD_III_IV}{Time to development of acute graft versus host disease stage III or IV}
#'  \item{survival_time numeric}{Part of surv::survival}
#'  \item{survival_status}{Part of surv::survival}
#' }
#' @name bone_marrow
#' @docType data
#' @title Medical data of periatric patients
#' @description The dataset describes pediatric patients with several hematologic diseases: malignant disorders
#' (i.a. patients with acute lymphoblastic leukemia, with acute myelogenous leukemia, with chronic myelogenous
#' leukemia, with myelodysplastic syndrome) and nonmalignant cases (i.a. patients with severe aplastic anemia,
#' with Fanconi anemia, with X-linked adrenoleukodystrophy).
#' All patients were subject to the unmanipulated allogeneic unrelated donor hematopoietic stem cell transplantation.
#' The motivation of this study was to identify the most important factors influencing the success or failure of the transplantation procedure.
#' In particular, verification of the research hypothesis that increased dosage of CD34+ cells / kg extends overall survival
#' time without simultaneous occurrence of undesirable events affecting patients' quality of life.
#' @format A data frame with 187 rows and 37 variables
#' @keywords datasets
NULL

#' @details Attribute information.
#' \itemize{
#'  \item{MM31}{numeric attribute}
#'  \item{MM116}{numeric attribute}
#'  \item{AS038}{numeric attribute}
#'  \item{PG072}{numeric attribute}
#'  \item{PD}{numeric attribute}
#'  \item{BA13}{numeric attribute}
#'  \item{DMM116}{numeric attribute}
#'  \item{MM116_pred}{label attribute}
#'  }
#' @name methane_train
#' @docType data
#' @title Training data for methane levle prediction
#' @descriptionTest Training data for methane levle prediction
#' @format A data frame with 13368 rows and 8 variables
#' @keywords datasets
NULL

#' @details Attribute information.
#' \itemize{
#'  \item{MM31}{numeric attribute}
#'  \item{MM116}{numeric attribute}
#'  \item{AS038}{numeric attribute}
#'  \item{PG072}{numeric attribute}
#'  \item{PD}{numeric attribute}
#'  \item{BA13}{numeric attribute}
#'  \item{DMM116}{numeric attribute}
#'  \item{MM116_pred}{label attribute}
#'  }
#' @name methane_test
#' @docType data
#' @title Testing data for methane levle prediction
#' @descriptionTest Testing data for methane levle prediction
#' @format A data frame with 5728 rows and 8 variables
#' @keywords datasets
NULL

