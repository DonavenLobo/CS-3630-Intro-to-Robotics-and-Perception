from main import *

if __name__ == '__main__':
    args = create_parser().parse_args()
    args.visualize = True

    p5 = Project5(args)
    successes, steps = p5.run()
    print(f'{successes} scenarios completed successfully')
