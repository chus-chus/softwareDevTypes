import pandas as pd
import numpy as np
from datetime import datetime

###############
# SELECT DATA #
###############

print("Selecting attributes...")

# GIT_COMMITS
gitCommits = pd.read_csv("../../data/raw/GIT_COMMITS.csv")

attributes = ['projectID', 'commitHash', 'author', 'committer', 'committerDate']
gitCommits = gitCommits[attributes]

gitCommits.to_csv('../../data/interim/DataPreparation/SelectData/GIT_COMMITS_select.csv', header=True)

# GIT_COMMITS_CHANGES
gitCommitsChanges = pd.read_csv("../../data/raw/GIT_COMMITS_CHANGES.csv")

attributes = ['projectID', 'commitHash', 'changeType', 'linesAdded', 'linesRemoved']
gitCommitsChanges = gitCommitsChanges[attributes]

gitCommitsChanges.to_csv('../../data/interim/DataPreparation/SelectData/GIT_COMMITS_CHANGES_select.csv', header=True)

# JIRA_ISSUES
jiraIssues = pd.read_csv("../../data/raw/JIRA_ISSUES.csv")

attributes = ['projectID', 'key', 'creationDate', 'resolutionDate', 'type', 'priority', 'assignee', 'reporter']
jiraIssues = jiraIssues[attributes]

jiraIssues.to_csv('../../data/interim/DataPreparation/SelectData/JIRA_ISSUES_select.csv', header=True)

# REFACTORING_MINER
refactoringMiner = pd.read_csv("../../data/raw/REFACTORING_MINER.csv")

attributes = ['projectID', 'commitHash', 'refactoringType']
refactoringMiner = refactoringMiner[attributes]

refactoringMiner.to_csv('../../data/interim/DataPreparation/SelectData/REFACTORING_MINER_select.csv', header=True)

# SONAR_ISSUES
sonarIssues = pd.read_csv("../../data/raw/SONAR_ISSUES.csv")

attributes = ['projectID', 'creationDate', 'closeDate', 'creationCommitHash', 'closeCommitHash', 'type', 'severity',
               'debt', 'author']
sonarIssues = sonarIssues[attributes]

sonarIssues.to_csv('../../data/interim/DataPreparation/SelectData/SONAR_ISSUES_select.csv', header=True)

# SONAR_MEASURES
sonarMeasures = pd.read_csv("../../data/raw/SONAR_MEASURES.csv")

attributes = ['commitHash', 'projectID', 'functions', 'commentLinesDensity', 'complexity', 'functionComplexity', 'duplicatedLinesDensity',
              'violations', 'blockerViolations', 'criticalViolations', 'infoViolations', 'majorViolations', 'minorViolations', 'codeSmells',
              'bugs', 'vulnerabilities', 'cognitiveComplexity', 'ncloc', 'sqaleIndex', 'sqaleDebtRatio', 'reliabilityRemediationEffort', 'securityRemediationEffort']
sonarMeasures = sonarMeasures[attributes]

sonarMeasures.to_csv('../../data/interim/DataPreparation/SelectData/SONAR_MEASURES_select.csv', header=True)

# SZZ_FAULT_INDUCING_COMMITS
szzFaultInducingCommits = pd.read_csv("../../data/raw/SZZ_FAULT_INDUCING_COMMITS.csv")

attributes = ['projectID', 'faultFixingCommitHash', 'faultInducingCommitHash', 'key']
szzFaultInducingCommits = szzFaultInducingCommits[attributes]

szzFaultInducingCommits.to_csv('../../data/interim/DataPreparation/SelectData/SZZ_FAULT_INDUCING_COMMITS_select.csv', header=True)

print("Attributes selected.")

##############
# CLEAN DATA #
##############

print("Cleaning data...")

def intersection(l1, l2):
    temp = set(l2)
    l3 = [value for value in l1 if value in temp]
    return l3

def difference(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))

# GIT_COMMITS
gitCommits = pd.read_csv("../../data/interim/DataPreparation/SelectData/GIT_COMMITS_select.csv")

authorNan = list(np.where(gitCommits.author.isna()))[0]
committerNan = list(np.where(gitCommits.committer.isna()))[0]
inters = intersection(authorNan, committerNan)
gitCommits = gitCommits.drop(inters)

gitCommits.to_csv('../../data/interim/DataPreparation/CleanData/GIT_COMMITS_clean.csv', header=True)

# GIT_COMMITS_CHANGES
gitCommitsChanges = pd.read_csv("../../data/interim/DataPreparation/SelectData/GIT_COMMITS_CHANGES_select.csv").iloc[:,1:]
gitCommitsChanges.to_csv('../../data/interim/DataPreparation/CleanData/GIT_COMMITS_CHANGES_clean.csv', header=True)

# JIRA_ISSUES
jiraIssues = pd.read_csv("../../data/interim/DataPreparation/SelectData/JIRA_ISSUES_select.csv").iloc[:,1:]

resolutionDate_nan = list(np.where(jiraIssues.resolutionDate.isna()))[0]
jiraIssues_notresolved = jiraIssues.iloc[resolutionDate_nan,:]
gitCommits = pd.read_csv("../../data/interim/DataPreparation/SelectData/GIT_COMMITS_select.csv").iloc[:,[1,-1]]
lastTimestamp = gitCommits.groupby(['projectID']).max()
jiraIssues_notresolved = pd.merge(jiraIssues_notresolved, lastTimestamp, how='left', on='projectID')
jiraIssues_notresolved = jiraIssues_notresolved.iloc[:,[0,1,2,4,5,6,7,8]].rename(columns={'committerDate': 'resolutionDate'})
jiraIssues_resolved = jiraIssues.drop(resolutionDate_nan)
jiraIssues = pd.concat([jiraIssues_resolved, jiraIssues_notresolved], sort=False).sort_index().reset_index().iloc[:,1:]

priority_nan = list(np.where(jiraIssues.priority.isna()))[0]
jiraIssues = jiraIssues.drop(priority_nan)

assignee_nan = list(np.where(jiraIssues.assignee.isna()))[0]
jiraIssues.assignee = jiraIssues.assignee.fillna('not-assigned')

jiraIssues.to_csv('../../data/interim/DataPreparation/CleanData/JIRA_ISSUES_clean.csv', header=True)

# REFACTORING_MINER
refactoringMiner = pd.read_csv("../../data/interim/DataPreparation/SelectData/REFACTORING_MINER_select.csv")

commitHashNan = list(np.where(refactoringMiner.commitHash.isna()))[0]
refactoringTypeNan = list(np.where(refactoringMiner.refactoringType.isna()))[0]
inters = intersection(commitHashNan, refactoringTypeNan)
refactoringMiner = refactoringMiner.drop(inters)

refactoringMiner.to_csv('../../data/interim/DataPreparation/CleanData/REFACTORING_MINER_clean.csv', header=True)

# SONAR_ISSUES
sonarIssues = pd.read_csv("../../data/interim/DataPreparation/SelectData/SONAR_ISSUES_select.csv").iloc[:,1:]

closeDateNan = list(np.where(sonarIssues.closeDate.isna()))[0]
closeCommitHashNan = list(np.where(sonarIssues.closeCommitHash.isna()))[0]
debtNan = list(np.where(sonarIssues.debt.isna()))[0]
authorNan = list(np.where(sonarIssues.author.isna()))[0]

inter = intersection(closeDateNan, closeCommitHashNan)
diff = difference(closeCommitHashNan, closeDateNan)
debtNan = list(np.where(sonarIssues.debt.isna())[0])
sonarIssues = sonarIssues.drop(debtNan).reset_index()
sonarIssues = sonarIssues.fillna({'closeCommitHash': 'not-resolved'})
gitCommits = pd.read_csv("../../data/interim/DataPreparation/CleanData/GIT_COMMITS_clean.csv").iloc[:,1:]
lastTimestamp = gitCommits.loc[:,['projectID', 'committerDate']].groupby(['projectID']).max()
closeDateNan = list(np.where(sonarIssues.closeDate.isna()))[0]
sonarIssues_notresolved = sonarIssues.iloc[closeDateNan,:]
sonarIssues_notresolved = pd.merge(sonarIssues_notresolved, lastTimestamp, how='left', on='projectID')
sonarIssues_notresolved = sonarIssues_notresolved.loc[:,['projectID', 'creationDate', 'creationCommitHash', 'closeCommitHash', 'type', 'severity', 'debt', 'author', 'committerDate']].rename(columns={'committerDate': 'closeDate'})
sonarIssues_resolved = sonarIssues.drop(closeDateNan)
sonarIssues = pd.concat([sonarIssues_resolved, sonarIssues_notresolved], sort=False).sort_index().reset_index().iloc[:,1:]
sonarIssues.groupby(['author']).size().reset_index().rename(columns={0:'count'})
df1 = gitCommits[['commitHash', 'committer']]
df2 = (sonarIssues[['creationCommitHash', 'author']]).rename(columns={'creationCommitHash': 'commitHash'})
merge = pd.merge(df1, df2, on='commitHash', how='inner').drop_duplicates()
pairs = merge.groupby(['committer', 'author']).size().reset_index().rename(columns={0:'count'})
index1 = list(np.where(pairs.committer.value_counts()==1))[0]
committer_1 = (pairs.committer.value_counts())[index1].index
index2 = list(np.where(pairs.author.value_counts()==1))[0]
author_1 = (pairs.author.value_counts())[index2].index
index_author_1 = pairs.loc[pairs['author'].isin(author_1)].index
index_committer_1 = pairs.loc[pairs['committer'].isin(committer_1)].index
inter_pairs = intersection(index_author_1, index_committer_1)
pairs_unique = pairs.loc[inter_pairs]
commiters = list(pairs_unique.committer)
authors = list(pairs_unique.author)
merge2 = pd.merge(merge, pairs_unique, on='committer', how='inner')
merge2 = merge2[['commitHash', 'committer', 'author_y']].rename(columns={'author_y': 'author', 'commitHash': 'creationCommitHash'})
merge2 = merge2.drop_duplicates()
prova2 = merge2[['creationCommitHash', 'author']]
dictionary = prova2.set_index('creationCommitHash').T.to_dict('records')[0]
sonarIssues.author = sonarIssues.author.fillna(sonarIssues.creationCommitHash.map(dictionary))
sonarIssues = sonarIssues.dropna(subset=['author'])
sonarIssues = sonarIssues.iloc[:,1:]

sonarIssues.to_csv('../../data/interim/DataPreparation/CleanData/SONAR_ISSUES_clean.csv', header=True)

# SONAR_MEASURES
sonarMeasures = pd.read_csv("../../data/interim/DataPreparation/SelectData/SONAR_MEASURES_select.csv")
sonarMeasures.to_csv('../../data/interim/DataPreparation/CleanData/SONAR_MEASURES_clean.csv', header=True)

# SZZ_FAULT_INDUCING_COMMITS
szzFaultInducingCommits = pd.read_csv("../../data/interim/DataPreparation/SelectData/SZZ_FAULT_INDUCING_COMMITS_select.csv").iloc[:,1:]
szzFaultInducingCommits.to_csv('../../data/interim/DataPreparation/CleanData/SZZ_FAULT_INDUCING_COMMITS_clean.csv', header=True)

print("Data cleaned.")

##################
# CONSTRUCT DATA #
##################

print("Constructing data...")

def produce_bug(x):
  if pd.isna(x.faultFixingCommitHash):
    return False
  return True

# COMMITS_FREQUENCY
gitCommits = pd.read_csv("../../data/interim/DataPreparation/CleanData/GIT_COMMITS_clean.csv")

gitCommits = gitCommits[['committer', 'committerDate']]
newDFsorted = gitCommits.sort_values(by=['committer', 'committerDate']).reset_index()[['committer', 'committerDate']]

newDFsortedCopy = []
committer = newDFsorted.iloc[0,0]
for index, row in newDFsorted.iterrows():
  if index != 0:
    if committer == newDFsorted.iloc[index,0]:
      r = (pd.to_datetime(newDFsorted.iloc[index,1])-pd.to_datetime(newDFsorted.iloc[index-1,1]))
      newDFsortedCopy.append([committer, r])
    else:
      committer = newDFsorted.iloc[index,0]

time_between_commits = pd.DataFrame(newDFsortedCopy)
time_between_commits[1] = time_between_commits[1].dt.total_seconds()

time_between_commits_commiter = time_between_commits.groupby([0]).mean()
time_between_commits_commiter = pd.DataFrame(time_between_commits_commiter).rename(columns={0:'committer', 1:'time_between_commits'})

time_between_commits_commiter.to_csv('../../data/interim/DataPreparation/ConstructData/COMMITS_FREQUENCY.csv', header=True)

# FIXED_ISSUES
gitCommits = pd.read_csv("../../data/interim/DataPreparation/CleanData/GIT_COMMITS_clean.csv")
SZZcommits = pd.read_csv("../../data/interim/DataPreparation/CleanData/SZZ_FAULT_INDUCING_COMMITS_clean.csv")
sonarIssues = pd.read_csv("../../data/interim/DataPreparation/CleanData/SONAR_ISSUES_clean.csv")
jiraIssues = pd.read_csv("../../data/interim/DataPreparation/CleanData/JIRA_ISSUES_clean.csv")

SZZcommits = SZZcommits['faultFixingCommitHash']
gitCommits = gitCommits[['commitHash', 'committer']]
sonarIssues = sonarIssues['closeCommitHash']
jiraIssues = jiraIssues['assignee']

SZZ_issues = (pd.merge(SZZcommits, gitCommits, how='inner', left_on='faultFixingCommitHash', right_on='commitHash').drop_duplicates())[['commitHash', 'committer']]
SSZ_issue_committer = SZZ_issues.committer.value_counts().rename_axis('committer').reset_index(name='SZZIssues')
Sonar_issues = pd.merge(sonarIssues, gitCommits, how='inner', left_on='closeCommitHash', right_on='commitHash').drop_duplicates()[['commitHash', 'committer']]
Sonar_issues_committer = Sonar_issues.committer.value_counts().rename_axis('committer').reset_index(name='SonarIssues')
Jira_issues_committer = jiraIssues[jiraIssues != 'not-assigned'].value_counts().rename_axis('committer').reset_index(name='JiraIssues')

issues = pd.merge(SSZ_issue_committer, Sonar_issues_committer, on='committer', how='outer')
issues = pd.merge(issues, Jira_issues_committer, on='committer', how='outer')
issues = issues.fillna(0)

issues.to_csv('../../data/interim/DataPreparation/ConstructData/FIXED_ISSUES.csv', header=True)

# INDUCED_ISSUES
gitCommits = pd.read_csv("../../data/interim/DataPreparation/CleanData/GIT_COMMITS_clean.csv")
SZZcommits = pd.read_csv("../../data/interim/DataPreparation/CleanData/SZZ_FAULT_INDUCING_COMMITS_clean.csv")
sonarIssues = pd.read_csv("../../data/interim/DataPreparation/CleanData/SONAR_ISSUES_clean.csv")

SZZcommits = SZZcommits['faultInducingCommitHash']
gitCommits = gitCommits[['commitHash', 'committer']]
sonarIssues = sonarIssues['creationCommitHash']

SZZ_issues = (pd.merge(SZZcommits, gitCommits, how='inner', left_on='faultInducingCommitHash', right_on='commitHash').drop_duplicates())[['commitHash', 'committer']]
SSZ_issue_committer = SZZ_issues.committer.value_counts().rename_axis('committer').reset_index(name='SZZIssues')
Sonar_issues = pd.merge(sonarIssues, gitCommits, how='inner', left_on='creationCommitHash', right_on='commitHash').drop_duplicates()[['commitHash', 'committer']]
Sonar_issues_committer = Sonar_issues.committer.value_counts().rename_axis('committer').reset_index(name='SonarIssues')

issues = pd.merge(SSZ_issue_committer, Sonar_issues_committer, on='committer', how='outer')
issues = issues.fillna(0)

issues.to_csv('../../data/interim/DataPreparation/ConstructData/INDUCED_ISSUES.csv', header=True)

# JIRA_ISSUES_time
jiraIssues = pd.read_csv("../../data/interim/DataPreparation/CleanData/JIRA_ISSUES_clean.csv").iloc[:,1:]

jiraIssues['creationDate'] =  pd.to_datetime(jiraIssues['creationDate'], format="%Y-%m-%dT%H:%M:%S.%f")
jiraIssues['resolutionDate'] =  pd.to_datetime(jiraIssues['resolutionDate'], format="%Y-%m-%dT%H:%M:%S.%f")
jiraIssues["resolutionTime"] = jiraIssues["resolutionDate"]

seconds = (jiraIssues.loc[:,"resolutionDate"] - jiraIssues.loc[:,"creationDate"]).dt.total_seconds()
jiraIssues.loc[:,"resolutionTime"] = seconds/3600

jiraIssues.to_csv('../../data/interim/DataPreparation/ConstructData/JIRA_ISSUES_time.csv', header=True)

# NUMBER_COMMITS
gitCommits = pd.read_csv("../../data/interim/DataPreparation/CleanData/GIT_COMMITS_clean.csv").iloc[:,2:]

number_commits = gitCommits.groupby(['committer']).count().iloc[1:,1]
number_commits = pd.DataFrame(number_commits).rename(columns={'commitHash': 'numberCommits'})

number_commits.to_csv('../../data/interim/DataPreparation/ConstructData/NUMBER_COMMITS.csv', header=True)

# REFACTORING_MINER_bug
refactoringMiner = pd.read_csv("../../data/interim/DataPreparation/CleanData/REFACTORING_MINER_clean.csv")[['projectID', 'commitHash', 'refactoringType']]
szzFaultInducingCommits = pd.read_csv("../../data/interim/DataPreparation/CleanData/SZZ_FAULT_INDUCING_COMMITS_clean.csv")[['projectID', 'faultFixingCommitHash', 'faultInducingCommitHash']]

induced_bug = pd.merge(refactoringMiner, szzFaultInducingCommits, how='left', left_on='commitHash', right_on='faultInducingCommitHash').drop_duplicates().reset_index()
induced_bug['bug'] = induced_bug.apply(lambda x: produce_bug(x), axis=1)
induced_bug = induced_bug[['projectID_x', 'commitHash', 'refactoringType', 'bug']].rename(columns={'projectID_x': 'projectID'})

induced_bug.to_csv('../../data/interim/DataPreparation/ConstructData/REFACTORING_MINER_bug.csv', header=True)

# SONAR_ISSUES_time
sonarIssues = pd.read_csv("../../data/interim/DataPreparation/CleanData/SONAR_ISSUES_clean.csv").iloc[:,1:]

sonarIssues['creationDate'] =  pd.to_datetime(sonarIssues['creationDate'], format='%Y-%m-%dT%H:%M:%SZ')
sonarIssues["closeDate"] =  pd.to_datetime(sonarIssues["closeDate"], format="%Y-%m-%dT%H:%M:%SZ")

sonarIssues["closeTime"] = sonarIssues["closeDate"]

seconds = (sonarIssues.loc[:,"closeDate"] - sonarIssues.loc[:,"creationDate"]).dt.total_seconds()
sonarIssues.loc[:,"closeTime"] = seconds/3600

sonarIssues.to_csv('../../data/interim/DataPreparation/ConstructData/SONAR_ISSUES_time.csv', header=True)

# SONAR_MEASURES_difference
sonarMeasures = pd.read_csv("../../data/interim/DataPreparation/CleanData/SONAR_MEASURES_clean.csv").iloc[:, 2:]
gitCommits = pd.read_csv("../../data/interim/DataPreparation/CleanData/GIT_COMMITS_clean.csv").iloc[:, 2:]

gitCommits['committerDate'] =  pd.to_datetime(gitCommits['committerDate'], format='%Y-%m-%dT%H:%M:%SZ')
newDF = pd.merge(sonarMeasures, gitCommits,  how='left', left_on=['commitHash','projectID'], right_on = ['commitHash','projectID'])
newDFNaN = list(np.where(newDF.committerDate.isna()))[0]
newDF = newDF.drop(newDFNaN)
projectID = newDF.projectID.unique()
newDFsorted = newDF.sort_values(by=['projectID', 'committerDate'])

newDFsortedCopy = newDFsorted.copy()
project = newDFsorted.iloc[0,1]
for index, row in newDFsorted.iterrows():
  if index < 55625:
    if project == newDFsorted.iloc[index,1]:
      r = newDFsortedCopy.iloc[index-1:index+1,2:22].diff().iloc[1,:]
      newDFsorted.iloc[index:index+1,2:22] = np.array(r)
    else:
      project = newDFsorted.iloc[index,1]

sonarMeasuresDifference = newDFsorted.iloc[:,:22]

sonarMeasuresDifference.to_csv('../../data/interim/DataPreparation/ConstructData/SONAR_MEASURES_difference.csv', header=True)

# TIME_IN_EACH_PROJECT
gitCommits = pd.read_csv("../../data/interim/DataPreparation/CleanData/GIT_COMMITS_clean.csv")

time_in_project = gitCommits.groupby(['projectID', 'committer'])['committerDate'].agg(['min', 'max']).reset_index()
time = (pd.to_datetime(time_in_project['max'])-pd.to_datetime(time_in_project['min']))
time_in_project['time'] = time.dt.total_seconds()
time_in_project = time_in_project[['projectID', 'committer', 'time']]

time_in_project.to_csv('../../data/interim/DataPreparation/ConstructData/TIME_IN_PROJECT.csv', header=True)

print("Data constructed.")

##################
# INTEGRATE DATA #
##################

print("Integarting data...")

numberCommits = pd.read_csv("../../data/interim/DataPreparation/ConstructData/NUMBER_COMMITS.csv")

fixedIssues = pd.read_csv("../../data/interim/DataPreparation/ConstructData/FIXED_ISSUES.csv").iloc[:,1:]
fixedIssues = fixedIssues.rename(columns={'SZZIssues':'fixedSZZIssues','SonarIssues':'fixedSonarIssues','JiraIssues':'fixedJiraIssues'})

inducedIssues = pd.read_csv("../../data/interim/DataPreparation/ConstructData/INDUCED_ISSUES.csv").iloc[:,1:]
inducedIssues = inducedIssues.rename(columns={'SZZIssues':'inducedSZZIssues','SonarIssues':'inducedSonarIssues'})
dataFrame = pd.merge(numberCommits, fixedIssues,  how='outer', left_on=['committer'], right_on = ['committer'])
dataFrame = dataFrame.fillna(0.0)
dataFrame = pd.merge(dataFrame, inducedIssues,  how='outer', left_on=['committer'], right_on = ['committer'])
dataFrame = dataFrame.fillna(0)

timeInProject = pd.read_csv("../../data/interim/DataPreparation/ConstructData/TIME_IN_PROJECT.csv").iloc[:,1:]
timeInProject = timeInProject.rename(columns={'time':'timeInProject'})
timeInProject = timeInProject.groupby(['committer']).mean().iloc[1:,:]
dataFrame = pd.merge(dataFrame, timeInProject,  how='outer', left_on=['committer'], right_on = ['committer'])
dataFrame = dataFrame.fillna(0)

jiraIssues = pd.read_csv("../../data/interim/DataPreparation/ConstructData/JIRA_ISSUES_time.csv").iloc[:,1:]
dum = pd.get_dummies(jiraIssues[["type", 'priority']], prefix=['type', 'priority'])
TypePriority = jiraIssues[['assignee']].join(dum)
TypePriority = TypePriority[TypePriority.assignee!='not-assigned'].reset_index().iloc[:,1:]
TypePriority = TypePriority.groupby(["assignee"]).sum()
resolutionTime = jiraIssues.loc[:,['assignee','resolutionTime']]
resolutionTime = resolutionTime.groupby(["assignee"]).mean()
jiraIssues = resolutionTime.join(TypePriority)
jiraIssues = jiraIssues.reset_index().rename(columns={'assignee':'committer'})
dataFrame = pd.merge(dataFrame, jiraIssues, how='left', left_on=['committer'], right_on = ['committer'])
dataFrame = dataFrame.fillna(0.0)

gitCommitsChanges = pd.read_csv("../../data/interim/DataPreparation/CleanData/GIT_COMMITS_CHANGES_clean.csv").iloc[:,2:]
dum = pd.get_dummies(gitCommitsChanges[["changeType"]])
dum = dum.rename(columns={'changeType_ModificationType.ADD':'ADD', 'changeType_ModificationType.DELETE':'DELETE', 'changeType_ModificationType.MODIFY':'MODIFY', 'changeType_ModificationType.RENAME':'RENAME', 'changeType_ModificationType.UNKNOWN':'UNKNOWN'})
Lines = gitCommitsChanges[["commitHash",'linesAdded','linesRemoved']]
gitCommitsChanges = pd.concat([Lines,dum], axis=1)
gitCommitsChanges = gitCommitsChanges.groupby(['commitHash']).agg({'ADD':'sum',	'DELETE':'sum',	'MODIFY':'sum',	'RENAME':'sum',	'UNKNOWN':'sum', 'linesAdded':'mean', 'linesRemoved':'mean'})
gitCommits = pd.read_csv("../../data/interim/DataPreparation/CleanData/GIT_COMMITS_clean.csv")
gitCommitsChanges = pd.merge(gitCommits, gitCommitsChanges, how='left', left_on=['commitHash'], right_on = ['commitHash'])
gitCommitsChanges = gitCommitsChanges[['committer','ADD','DELETE','MODIFY','RENAME','UNKNOWN','linesAdded','linesRemoved']]
gitCommitsChanges = gitCommitsChanges.groupby(['committer']).mean()
dataFrame = pd.merge(dataFrame, gitCommitsChanges, how='left', left_on=['committer'], right_on = ['committer'])
dataFrame = dataFrame.fillna(0.0)

refactoringMinerBug = pd.read_csv("../../data/interim/DataPreparation/ConstructData/REFACTORING_MINER_bug.csv").iloc[:,1:]
dum = pd.get_dummies(refactoringMinerBug[['refactoringType', 'bug']])
commitHash = refactoringMinerBug[["commitHash"]]
refactoringMinerBug = pd.concat([commitHash,dum], axis=1)
refactoringMinerBug = refactoringMinerBug.groupby(['commitHash']).sum()
refactoringMinerBug = pd.merge(refactoringMinerBug, gitCommits, how='left', left_on=['commitHash'], right_on = ['commitHash'])
refactoringMinerBug = pd.concat([refactoringMinerBug[['committer']], refactoringMinerBug.iloc[:,:-4]], axis=1)
refactoringMinerBug = refactoringMinerBug.groupby(['committer']).sum()
dataFrame = pd.merge(dataFrame, refactoringMinerBug, how='left', left_on=['committer'], right_on = ['committer'])
dataFrame = dataFrame.fillna(0.0)

sonarMeasures = pd.read_csv("../../data/interim/DataPreparation/ConstructData/SONAR_MEASURES_difference.csv").iloc[:,1:]
gitCommits = pd.read_csv("../../data/interim/DataPreparation/CleanData/GIT_COMMITS_clean.csv")[['commitHash', 'committer']]
sonarMeasures = pd.merge(sonarMeasures, gitCommits, how='left', on='commitHash').iloc[:,2:]
sonarMeasures_committer = sonarMeasures.groupby(['committer']).agg({'functions':'sum', 'commentLinesDensity':'mean',
'complexity':'sum', 'functionComplexity':'sum', 'duplicatedLinesDensity':'mean', 'violations':'sum', 'blockerViolations':'sum',
 'criticalViolations':'sum','infoViolations':'sum','majorViolations':'sum','minorViolations':'sum','codeSmells':'sum',
 'bugs':'sum','vulnerabilities':'sum','cognitiveComplexity':'sum','ncloc':'sum','sqaleIndex':'sum',
 'sqaleDebtRatio':'sum','reliabilityRemediationEffort':'sum','securityRemediationEffort':'sum'}).reset_index()
dataFrame = pd.merge(dataFrame, sonarMeasures_committer, how='left', on='committer')
dataFrame = dataFrame.fillna(0)

sonarIssues = pd.read_csv("../../data/interim/DataPreparation/ConstructData/SONAR_ISSUES_time.csv").iloc[:,4:]
sonarIssues = pd.concat([sonarIssues[['creationCommitHash']], sonarIssues.iloc[:,2:-2], sonarIssues[['closeTime']]], axis=1)
debtSec = sonarIssues.debt.apply(pd.Timedelta)
debtHour = debtSec.apply(lambda x: x.seconds/3600 + x.days*24)
sonarIssues[['debt']] = debtHour.to_frame()
dum = pd.get_dummies(sonarIssues[['type','severity']])
sonarIssues = pd.concat([sonarIssues[['creationCommitHash','debt','closeTime']], dum], axis=1)
closeTime = sonarIssues.loc[:,['creationCommitHash','closeTime']]
closeTime = closeTime.groupby(["creationCommitHash"]).mean()
Debt = sonarIssues[['creationCommitHash','debt']]
Debt = Debt.groupby(['creationCommitHash']).sum()
TypesSeverities = pd.concat([sonarIssues[['creationCommitHash']], sonarIssues.iloc[:,3:]], axis=1)
TypesSeverities = TypesSeverities.groupby(['creationCommitHash']).sum()
sonarIssues = pd.concat([Debt, closeTime, TypesSeverities], axis=1)
sonarIssues2 = pd.merge(sonarIssues, gitCommits, how='left', left_on=['creationCommitHash'], right_on=['commitHash'])
closeTime = sonarIssues2.loc[:,['committer','closeTime']]
closeTime = closeTime.groupby(["committer"]).mean()
Debt = sonarIssues2[['committer','debt']]
Debt = Debt.groupby(['committer']).sum()
TypesSeverities = pd.concat([sonarIssues2[['committer']], sonarIssues2.iloc[:,2:-3]], axis=1)
TypesSeverities = TypesSeverities.groupby(['committer']).sum()
sonarIssues2 = pd.concat([Debt, closeTime, TypesSeverities], axis=1)
sonarIssues2['committer'] = sonarIssues2.index
sonarIssues2 = sonarIssues2.iloc[:,:-1]
dataFrame = pd.merge(dataFrame, sonarIssues2, how='left', on='committer')
dataFrame = dataFrame.fillna(0.0)

dataFrame.to_csv('../../data/interim/DataPreparation/DATA_FRAME.csv', header=True)

print("Data integrated.")

###############
# FORMAT DATA #
###############

dataFrame = pd.read_csv("../../data/interim/DataPreparation/DATA_FRAME.csv").iloc[:,1:]

dic = {"time_between_commits":"timeBetweenCommits", 'type_Bug':	'jiraBug', 'type_Dependency upgrade':'jiraDependencyUpgrade',
       'type_Documentation':'jiraDocumentation',	'type_Epic':'jiraEpic', 'type_Improvement':'jiraImprovement',
       'type_New Feature':'jiraNewFeature',	'type_Question':'jiraQuestion',	'type_Story':'jiraStory',
       'type_Sub-task':'jiraSub-task',	'type_Task':'jiraTask',	'type_Technical task':'jiraTechnicalTask',
       'type_Test':'jiraTest',	'type_Wish':'jiraWish', 'priority_Blocker':'jiraBlocker',	'priority_Critical':'jiraCritical',
       'priority_Major':'jiraMajor',	'priority_Minor':'jiraMinor',	'priority_Trivial':'jiraTrivial',
       'ADD':'commitChangeAdd',	'DELETE':'commitChangeDelete',	'MODIFY':'commitChangeModify',	'RENAME':'commitChangeRename',
       'UNKNOWN':'commitChangeUnknown', 'bug':'refactoringInducedBug', 'refactoringType_Change Package':'refactoringChangePackage',
       'refactoringType_Extract And Move Method':'refactoringExtractAndMoveMethod', 'refactoringType_Extract Class':'refactoringExtractClass',
       'refactoringType_Extract Interface':'refactoringExtractInterface', 'refactoringType_Extract Method':'refactoringExtractMethod',
       'refactoringType_Extract Subclass':'refactoringExtractSubclass', 'refactoringType_Extract Superclass':'refactoringExtractSuperclass',
       'refactoringType_Extract Variable':'refactoringExtractVariable', 'refactoringType_Inline Method':'refactoringInlineMethod',
       'refactoringType_Inline Variable':'refactoringInlineVariable', 'refactoringType_Move And Rename Attribute':'refactoringMoveAndRenameAttribute',
       'refactoringType_Move And Rename Class':'refactoringMoveAndRenameClass', 'refactoringType_Move Attribute':'refactoringMoveAttribute',
       'refactoringType_Move Class':'refactoringMoveClass', 'refactoringType_Move Method':'refactoringMoveMethod',
       'refactoringType_Move Source Folder':'refactoringMoveSourceFolder', 'refactoringType_Parameterize Variable':'refactoringParameterizeVariable',
       'refactoringType_Pull Up Attribute':'refactoringPullUpAttribute', 'refactoringType_Pull Up Method':'refactoringPullUpMethod',
       'refactoringType_Push Down Attribute':'refactoringPushDownAttribute', 'refactoringType_Push Down Method':'refactoringPushDownMethod',
       'refactoringType_Rename Attribute':'refactoringRenameAttribute', 'refactoringType_Rename Class':'refactoringRenameClass',
       'refactoringType_Rename Method':'refactoringRenameMethod', 'refactoringType_Rename Package':'refactoringRenamePackage',
       'refactoringType_Rename Parameter':'refactoringRenameParameter', 'refactoringType_Rename Variable':'refactoringRenameVariable',
       'refactoringType_Replace Attribute':'refactoringReplaceAttribute', 'refactoringType_Replace Variable With Attribute':'refactoringReplaceVariableWithAttribute',
       'functions':'codeFunctions', 'commentLinesDensity':'codeCommentLinesDensity', 'complexity':'codeComplexity', 'functionComplexity':'codeFunctionComplexity',
       'duplicatedLinesDensity':'codeDuplicatedLinesDensity','violations':'codeViolations', 'blockerViolations':'codeBlockerViolations',
       'criticalViolations':'codeCriticalViolations', 'infoViolations':'codeInfoViolations', 'majorViolations':'codeMajorViolations',
       'minorViolations':'codeMinorViolations', 'codeSmells':'codeCodeSmells', 'bugs':'codeBugs', 'vulnerabilities':'codeVulnerabilities',
       'cognitiveComplexity':'codeCognitiveComplexity', 'ncloc':'codeNcloc', 'sqaleIndex':'codeSqaleIndex', 'sqaleDebtRatio':'codeSqaleDebtRatio',
       'reliabilityRemediationEffort':'codeReliabilityRemediationEffort', 'securityRemediationEffort':'codeSecurityRemediationEffort',
       'debt':'sonarDebt', 'closeTime':'sonarCloseTime','type_BUG':'sonarBug', 'type_CODE_SMELL':'sonarCodeSmell', 'type_VULNERABILITY':'sonarVulnerability',
       'severity_BLOCKER':'sonarBlocker', 'severity_CRITICAL':'sonarCritical', 'severity_INFO':'sonarInfo', 'severity_MAJOR':'sonarMajor',
       'severity_MINOR':'sonarMinor'}

dataFrame = dataFrame.rename(columns=dic)
dataFrame.to_csv('../../data/processed/DEVELOPERS_DATA.csv', header=True)
