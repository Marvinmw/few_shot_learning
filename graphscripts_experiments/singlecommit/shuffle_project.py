import random
import json
projects = ["Cli", "Chart", "Closure", "Compress", "Csv", "Gson", "JacksonCore", "JacksonDetabind", "JacksonXml", "Jsoup", "JxPath", "Lang", "Math", "Mockito", "Time"]
random.shuffle( projects )
pre_projects = projects[:int(len(projects)*0.6)]
remaining_projects = projects[int(len(projects)*0.6):]
json.dump( pre_projects, open("pre_project.json", "w") )
json.dump( remaining_projects, open("remaining_projects.json", "w") )