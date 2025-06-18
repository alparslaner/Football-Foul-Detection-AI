import os
import torch
import json
from config.classes import EVENT_DICTIONARY

##############################################################
#                                                            #
#              DO NOT MAKE ANY CHANGES HERE                  #
#                                                            #
##############################################################



# Function to load the labels from the json file
def label2vectormerge(folder_path, split, num_views):
	path_annotations = os.path.join(folder_path, split)
	path_annotations = os.path.join(path_annotations, "annotations.json") 

	if os.path.exists(path_annotations):
		with open(path_annotations) as f:
			train_annotations_data = json.load(f)
	else:
		print("PATH DOES NOT EXISTS")
		exit()

	not_taking = []

	# Define number of classes for each feature
	num_classes_bodypart = 2
	num_classes_action = 8
	num_classes_multiple = 2
	num_classes_tryplay = 2
	num_classes_touchball = 2
	num_classes_goalpos = 3
	num_classes_severity = 6

	# Initialize label lists
	labels_bodypart = []
	labels_action = []
	labels_multiple = []
	labels_tryplay = []
	labels_touchball = []
	labels_goalpos = []
	labels_severity = []
	number_of_actions = []

	# Initialize distribution tensors
	distribution_bodypart = torch.zeros(1, num_classes_bodypart)
	distribution_action = torch.zeros(1, num_classes_action)
	distribution_multiple = torch.zeros(1, num_classes_multiple)
	distribution_tryplay = torch.zeros(1, num_classes_tryplay)
	distribution_touchball = torch.zeros(1, num_classes_touchball)
	distribution_goalpos = torch.zeros(1, num_classes_goalpos)
	distribution_severity = torch.zeros(1, num_classes_severity)

	if num_views == 1:
		for actions in train_annotations_data['Actions']:
			action_class = train_annotations_data['Actions'][actions]['Action class']
			bodypart = train_annotations_data['Actions'][actions]['Body part']
			multiple = train_annotations_data['Actions'][actions]['Multiple fouls']
			tryplay = train_annotations_data['Actions'][actions]['Try to play']
			touchball = train_annotations_data['Actions'][actions]['Touch ball']
			goalpos = train_annotations_data['Actions'][actions]['Close goal position']
			severity = train_annotations_data['Actions'][actions]['Severity']

			# Convert labels to one-hot vectors
			labels_bodypart.append(torch.zeros(1, num_classes_bodypart))
			labels_bodypart[-1][0][bodypart] = 1
			distribution_bodypart[0][bodypart] += 1

			labels_action.append(torch.zeros(1, num_classes_action))
			labels_action[-1][0][action_class] = 1
			distribution_action[0][action_class] += 1

			labels_multiple.append(torch.zeros(1, num_classes_multiple))
			labels_multiple[-1][0][multiple] = 1
			distribution_multiple[0][multiple] += 1

			labels_tryplay.append(torch.zeros(1, num_classes_tryplay))
			labels_tryplay[-1][0][tryplay] = 1
			distribution_tryplay[0][tryplay] += 1

			labels_touchball.append(torch.zeros(1, num_classes_touchball))
			labels_touchball[-1][0][touchball] = 1
			distribution_touchball[0][touchball] += 1

			labels_goalpos.append(torch.zeros(1, num_classes_goalpos))
			labels_goalpos[-1][0][goalpos] = 1
			distribution_goalpos[0][goalpos] += 1

			labels_severity.append(torch.zeros(1, num_classes_severity))
			labels_severity[-1][0][severity] = 1
			distribution_severity[0][severity] += 1

	return (labels_bodypart, labels_action, labels_multiple, labels_tryplay,
			labels_touchball, labels_goalpos, labels_severity,
			distribution_bodypart, distribution_action, distribution_multiple,
			distribution_tryplay, distribution_touchball, distribution_goalpos,
			distribution_severity)


# Function to load the path to the clips
def clips2vectormerge(folder_path, split, num_views, not_taking):

	path_clips = os.path.join(folder_path, split)

	if os.path.exists(path_clips):
		folders = 0

		for _, dirnames, _ in os.walk(path_clips):
			folders += len(dirnames) 
			
		clips = []
		for i in range(folders):
			if str(i) in not_taking:
				continue
			
			if num_views == 1:
				path_clip = os.path.join(path_clips, "action_" + str(i))
				path_clip_0 = os.path.join(path_clip, "clip_0.mp4")
				clips_all_view = []
				clips_all_view.append(path_clip_0)
				clips.append(clips_all_view)
				clips_all_view = []
				path_clip_1 = os.path.join(path_clip, "clip_1.mp4")
				clips_all_view.append(path_clip_1)
				clips.append(clips_all_view)
				clips_all_view = []

				if os.path.exists(os.path.join(path_clip, "clip_2.mp4")):
					path_clip_2 = os.path.join(path_clip, "clip_2.mp4")
					clips_all_view.append(path_clip_2)
					clips.append(clips_all_view)
					clips_all_view = []

				if os.path.exists(os.path.join(path_clip, "clip_3.mp4")):
					path_clip_3 = os.path.join(path_clip, "clip_3.mp4")
					clips_all_view.append(path_clip_3)
					clips.append(clips_all_view)
					clips_all_view = []
			else:
				path_clip = os.path.join(path_clips, "action_" + str(i))
				path_clip_0 = os.path.join(path_clip, "clip_0.mp4")
				clips_all_view = []
				clips_all_view.append(path_clip_0)
				path_clip_1 = os.path.join(path_clip, "clip_1.mp4")
				clips_all_view.append(path_clip_1)

				if os.path.exists(os.path.join(path_clip, "clip_2.mp4")):
					path_clip_2 = os.path.join(path_clip, "clip_2.mp4")
					clips_all_view.append(path_clip_2)

				if os.path.exists(os.path.join(path_clip, "clip_3.mp4")):
					path_clip_3 = os.path.join(path_clip, "clip_3.mp4")
					clips_all_view.append(path_clip_3)
				clips.append(clips_all_view)

		return clips


