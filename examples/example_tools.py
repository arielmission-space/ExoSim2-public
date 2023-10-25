import exosim.tools as tools
from exosim.utils import RunConfig

# this will force the use of all the cpu except 2
RunConfig.n_job = -2
RunConfig.random_seed = 10


def main():
    tools.QuantumEfficiencyMap(
        options_file="tools_input_example.xml", output="data/payload/qe_map.h5"
    )

    tools.ReadoutSchemeCalculator(
        options_file="tools_input_example.xml", input_file="test_common.h5"
    )

    tools.DeadPixelsMap(
        options_file="tools_input_example.xml", output="data/payload"
    )

    tools.PixelsNonLinearity(
        options_file="tools_input_example.xml",
        output="data/payload/pnl_map.h5",
        show_results=True,
    )


if __name__ == "__main__":
    main()
