import colossalai



if __name__=="__main__":

    #args = colossalai.get_default_parser().parse_args()

    # launch distributed environment
    colossalai.launch_from_torch(config='./config.py'
    )

    # colossalai.launch(config='./config.py',
    #               rank=0,
    #               world_size=1,
    #               host='127.0.0.1',
    #               port=631)



