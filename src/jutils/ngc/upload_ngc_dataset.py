import os
import argparse
import subprocess
import re
from coolname import generate_slug
import time
from datetime import timedelta

def parse_cmd_line():

	# python upload_ngc_dataset.py --f="./folder spaces/small Data.zip" -n="dl4rt_mini45MB_ds" -t=dt --desc="mini 45MB dataset"

	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--file', '-f', type=str, default=None, required=True,
		help='compressed file to upload'
	)
	parser.add_argument(
		'--dataset_name', '-n', type=str, default=None, required=True,
		help='name you want to give your final dataset'
	)
	parser.add_argument(
		'--team', '-t', type=str, default=None, required=True,
		help='name of the team for batch job'
	)
	parser.add_argument(
		'--desc', '-d', type=str, default=None, required=True,
		help='description of dataset'
	)
	parser.add_argument(
		'--image', '-i', type=str, default=None, required=False,
		help="Image that contains the extraction utilities (zip, tar, etc)"
	)
	parser.add_argument(
		'--check_job_freq', type=int, default=120, required=False,
		help="frequency in seconds to check the job status"
	)
	parser.add_argument(
		'--check_convert_freq', type=int, default=300, required=False,
		help="frequency in seconds to attempt conversion as we wait for results to be available"
	)
	parser.add_argument(
		'--keep_temp_dataset',  action='store_true', default=False, required=False,
		help='Does not delete the temporary dataset with the compressed file on completion'
	)
	parser.add_argument(
		'--custom_cmdline', type=str, default=None, required=False,
		help='Allows for custom command line.  Insert {} where the compressed file should go in the command.  Example: --custom_cmdline=\"unzip \"{}\" -d ./results/\"'
	)


	return parser


def upload_compressed_file(filename):
	# ngc dataset upload --source <zipfile> <datasetname> 

	temp_dataset_name = generate_slug(2)
	# os.system('ngc dataset upload --source {} {}'.format(filename, temp_dataset_name))
	upload_command_tokenized = "ngc dataset upload --source".split( ' ' )
	upload_command_tokenized.append( filename )
	upload_command_tokenized.append( temp_dataset_name )

	print( "Uploading compressed file to temp Dataset: {}".format( " ".join( upload_command_tokenized ) ) )

	proc = subprocess.Popen(upload_command_tokenized, stdout = subprocess.PIPE, shell=False)
	stdout, stderr = proc.communicate()

	# find the Dataset ID
	dataset_id = None
	match = re.search( "Dataset ID: (\d+)", str( stdout ) )
	if match:
		dataset_id = match.group(1)
		print( "Temporary Dataset: {}:{}".format( temp_dataset_name, dataset_id ) )
	else:
		fail_and_exit( stdout )

	return temp_dataset_name, dataset_id


def get_decompress_cmd(filename, custom_cmd):
	_, ext = os.path.splitext( filename )
	if custom_cmd is not None:
		return custom_cmd.format( "./data" + "/" + filename )
	elif ext == ".zip":
		return "unzip \"{}\" -d ./results/".format("./data" + "/" + filename)
	elif ext == ".tar":
		return "tar xvf \"{}\" -C ./results/".format("./data" + "/" + filename)
	elif ext == ".tgz":
		return "tar xvzf \"{}\" -C ./results/".format("./data" + "/" + filename)


def extract_compressed_file(filename, temp_dataset_name, team, dataset_id, custom_cmdline, custom_image ):
	'''
	ngc batch run 
		--name [batch_name]
		 --preempt RUNONCE 
		 --min-timeslice 0s 
		 --total-runtime 0s 
		 --ace nv-us-west-2 
		 --instance dgx1v.16g.1.norm 
		 --commandline [unzip command]
		 --result /workspace/results 
		 --image [image_name:tag]
		 --org nvidian 
		 --team [team]
		 --datasetid [dataset_id]:/workspace/data
	'''

	compressed_filename = os.path.basename( filename )

	# run an NGC batch with a command to unzip the data set
	extract_job_tokenized = "ngc batch run".split( ' ' )
	extract_job_tokenized.append( "--name" )
	extract_job_tokenized.append( "{}".format( "auto_extract_{}".format( temp_dataset_name ) ) ) 
	extract_job_tokenized.append( "--preempt" ) 
	extract_job_tokenized.append( "RUNONCE" )
	extract_job_tokenized.append( "--min-timeslice" )
	extract_job_tokenized.append( "0s" ) 
	extract_job_tokenized.append( "--total-runtime" )
	extract_job_tokenized.append( "0s" )
	extract_job_tokenized.append( "--ace" )
	extract_job_tokenized.append( "nv-us-west-2" )
	extract_job_tokenized.append( "--instance" )
	extract_job_tokenized.append( "dgx1v.16g.1.norm" ) 
	extract_job_tokenized.append( "--commandline" )
	# "unzip \"{}\" -d ./results/"
	# tar xvf ./data2/small\ Data.tar -C ./results/
	# tar xvzf ./data2/small\ data.tgz -C ./results
	#extract_job_tokenized.append( "unzip \"{}\" -d ./results/".format("./data" + "/" + compressed_filename)) 
	extract_job_tokenized.append( get_decompress_cmd( compressed_filename, custom_cmdline )) 
	extract_job_tokenized.append( "--result" )
	extract_job_tokenized.append( "/workspace/results" )
	extract_job_tokenized.append( "--image" )
	if custom_image is None:
		extract_job_tokenized.append( "nvidian/dt/dl4rt:pth1.4_py3.7_cuda10.1" )
	else:
		extract_job_tokenized.append( custom_image )
	extract_job_tokenized.append( "--org" )
	extract_job_tokenized.append( "nvidian" )
	extract_job_tokenized.append( "--team" )
	extract_job_tokenized.append( team )
	extract_job_tokenized.append( "--datasetid" )
	extract_job_tokenized.append( "{}:/workspace/data".format( dataset_id ) )

	extract_command_line = " ".join( extract_job_tokenized )
	print( "Executing NGC Batch job: {}".format( extract_command_line ) )

	proc = subprocess.Popen(extract_job_tokenized, stdout = subprocess.PIPE, shell=False)
	stdout, stderr = proc.communicate()

		# find the Job ID
	job_id = None
	match = re.search( "Id: (\d+)", str( stdout ) )
	if match:
		print( "Job ID: ", match.group(1) )
		job_id = match.group(1)
	else:
		fail_and_exit( stdout )

	return job_id


def wait_for_job_completion(job_id, job_check_freq):
	#  ngc batch info [job_id]

	print( "Waiting for Job {} to complete".format(job_id))

	job_status = None
	prior_job_status = None
	batch_info_tokenized = 'ngc batch info {}'.format(job_id).split( ' ' )
	while( not is_job_complete( job_status ) ):
		# job will probably take a while to start up and execute
		time.sleep( job_check_freq )

		proc = subprocess.Popen(batch_info_tokenized, stdout = subprocess.PIPE, shell=False)
		stdout, stderr = proc.communicate()

		match = re.search( "Status: (\w+)", str( stdout ) )
		if match:
			job_status = match.group(1)
			if job_status != prior_job_status:
				prior_job_status = job_status
				print( "Job Status: {}".format( job_status ))
		else:
			fail_and_exit( stdout )

	return job_status


def convert_to_dataset(job_id, desc, dataset_name, check_convert_freq):
	# ngc dataset convert --from-result [ job id ] --desc [desc] [dataset_name]

	convert_tokenized = "ngc dataset convert".split( ' ' )
	convert_tokenized.append("--from-result")
	convert_tokenized.append(job_id)
	convert_tokenized.append("--desc")
	convert_tokenized.append(desc)
	convert_tokenized.append(dataset_name)

	print( "Converting Dataset:  {}".format( ' '.join( convert_tokenized ) ) )

	new_dataset_id = None
	while ( new_dataset_id is None ):
		# job will probably take a while to start up and execute
		time.sleep( check_convert_freq )

		proc = subprocess.Popen(convert_tokenized, stdout = subprocess.PIPE, shell=False)
		stdout, stderr = proc.communicate()

		# print( stdout )

		waiting_match = re.search( "Resultset files not fully registered yet", str( stdout ) )
		if waiting_match:
			print("Results for job {} not ready yet.  Waiting...".format( job_id ) )
		else:
			converted_match = re.search( "Dataset with id: (\d+)", str( stdout ) )
			if converted_match:
				new_dataset_id = converted_match.group(1)

	return new_dataset_id


def remove_dataset(temp_dataset_name, dataset_id):
	# ngc dataset remove [dataset_id] -y
	print( "Removing temporary dataset {}".format(temp_dataset_name))
	os.system('ngc dataset remove {} -y'.format(dataset_id))
	os.system('ngc dataset list --owned' )


def compressed_file_to_dataset(args):
	start_time = time.time()

	temp_dataset_name, dataset_id = upload_compressed_file( args.file )
	job_id = extract_compressed_file( args.file, temp_dataset_name, args.team, dataset_id, args.custom_cmdline, args.image )
	job_status = wait_for_job_completion( job_id, args.check_job_freq )

	# now wait and in 5 minute intervals attempt to convert results to a dataset
	if job_status == "FINISHED_SUCCESS":
		print( "Successfully extracted job.  Converting results to dataset..." )
		new_dataset_id = convert_to_dataset( job_id, args.desc, args.dataset_name, args.check_convert_freq )
		print( "finished converting {} to dataset {}:{}".format(args.file, args.dataset_name, new_dataset_id))

	else:
		print( "job status ended in {}.  Your dataset was not extracted".format( job-status ) )

	if not args.keep_temp_dataset:
		remove_dataset( temp_dataset_name, dataset_id )

	elapsed_time = timedelta(seconds=time.time() - start_time)
	print( "\n\nElapsed Time = {}".format( td_format(elapsed_time) ) )


def fail_and_exit( final_output ):
	print("I have failed you.\n")
	print("I am unable to extract critical information. Here is the output\n\n" )
	print( final_output )
	print("listing owned datasets so that user can perform cleanup with 'ngc dataset remove <dataset id>'\n\n" )
	os.system('ngc dataset list --owned' )
	exit()


def is_job_complete( job_status ):
	return job_status == "FINISHED_SUCCESS" or \
		   job_status == "KILLED_BY_USER" or \
		   job_status == "KILLED_BY_SYSTEM" or \
		   job_status == "FAILED" or \
		   job_status == "FAILED_RUN_LIMIT_EXCEEDED" or \
		   job_status == "TASK_LOST" or \
		   job_status == "TASK_LOST" or \
		   job_status == "CANCELED"


# Utility function for pretty deltatime printing
def td_format(td_object):
    seconds = int(td_object.total_seconds())
    periods = [
        ('year',        60*60*24*365),
        ('month',       60*60*24*30),
        ('day',         60*60*24),
        ('hour',        60*60),
        ('minute',      60),
        ('second',      1)
    ]
 
    strings=[]
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            period_value , seconds = divmod(seconds, period_seconds)
            has_s = 's' if period_value > 1 else ''
            strings.append("%s %s%s" % (period_value, period_name, has_s))
 
    return ", ".join(strings)


def main():
	parser = parse_cmd_line()
	cmd_args = parser.parse_args()
	# create_dataset(cmd_args)
	compressed_file_to_dataset( cmd_args )

if __name__ == '__main__':
    main()