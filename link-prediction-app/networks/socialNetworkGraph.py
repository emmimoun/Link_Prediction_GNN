from networks.graph import Graph


class SocialNetworkGraph(Graph):
    def __init__(self, driver, config=None):
        super().__init__(driver, "UNDIRECTED", config)
