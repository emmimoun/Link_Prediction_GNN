
def show_databases(G, k=None, normalized=True, weight=None, endpoints=False, seed=None):
    # doesn't currently support `weight`, `k`, `endpoints`, `seed`

    query = """\
    show databases
    """ % G.identifier_property

    params = G.base_params()

    with G.driver.session() as session:
        result = {row["node"]: row["score"] for row in session.run(query, params)}
    return result

