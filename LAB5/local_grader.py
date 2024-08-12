import random
import argparse
import numpy as np
from main import *

def create_parser():
    parser = argparse.ArgumentParser(description='Run the Robatarium simulator using your PID controller :)')
    parser.add_argument('--scenario', '-s', default='all', type=str, help='The scenario to run, "all" to run all scenarios')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize the scenario')
    parser.add_argument('--real_time', '-r', action='store_true', help='Run sim at same speed as the real robot would run')
    return parser

args = create_parser().parse_args()

scores=[]
ideal_avg_steps=[480,480,480,500,550,550]
all_avg_steps = []

scenarios=[]
if args.scenario == 'all':
    scenarios = range(1,7)
else:
    scenarios = [int(args.scenario)]

for scenario in scenarios:
    args.scenario = str(scenario)
    p5=Project5(args)
    print("Running scenario ",scenario)
    all_results=[]
    seeds_passed=0
    steps=[]
    fail_count=0
    scenario_score=0
    flag=0
    for i in range(0,30):
    # for i in [10]:
        print(f'Seed: {i}')  
        random.seed(i)
        np.random.seed(i)
        result=p5.run()
        if result[0]:
            seeds_passed+=1
            steps.append(result[1][0])
        else:
            print("Scenario ", scenario," failed at seed ",i)
            fail_count+=1
        if fail_count>1:
            scenario_score=0
            flag=1
    print(seeds_passed, "/30 seeds passed for scenario ",scenario)
    if seeds_passed>=29:
        if seeds_passed==30:
            steps.remove(max(steps))
        avg_steps=sum(steps)/len(steps)
        print(avg_steps)
        all_avg_steps.append(avg_steps)
        if flag==0:
            if scenario == 1:
                scenario_score = min(10,(10 * min(1, (ideal_avg_steps[scenario-1]) / avg_steps)))
            else:
                scenario_score = min(18,(18 * min(1, (ideal_avg_steps[scenario-1] / avg_steps))))
    scores.append(round(scenario_score,3))
print("Average steps for each scenario:", all_avg_steps)
print("Final scores for each scenario:", scores)
print("Total score =",sum(scores))
