import exosim.recipes as recipes
from exosim.utils import RunConfig

# this will force the use of all the cpu except 2
RunConfig.n_job = -2
RunConfig.random_seed = 10


def main():
    # # run radiometric model
    recipes.RadiometricModel(
        "main_radiometric.xml", "./test_radio.h5", plot=True, isolate_every_opt=False
    )


if __name__ == "__main__":
    main()
