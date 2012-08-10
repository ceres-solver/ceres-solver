import os
import sys
import subprocess

problem_dir = "/Users/sameeragarwal/Downloads/bal-problems"
num_iterations = 50
cmd_template = "./examples/bundle_adjuster -problem_name=%s -performance_dir=%s --input=%s/%s -trust_region_strategy=%s -num_iterations=%s"
dogleg_dir = "/tmp/dogleg-30"
lm_dir = "/tmp/lm-30"
problems = os.listdir(problem_dir)

#os.mkdir(dogleg_dir)
#os.mkdir(lm_dir)

for problem in problems:
	cmd = cmd_template%(problem, dogleg_dir, problem_dir, problem, "dogleg", num_iterations)
	print "dogleg", problem
	print cmd
	status = subprocess.call(cmd, shell=True)
	if (status == 0):
		print "success"
	else:
		print "failure"
		break
