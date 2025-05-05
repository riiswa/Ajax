import distrax


class SquashedNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        self.tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=self.tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())

    def entropy(self):
        return self.distribution.entropy()
