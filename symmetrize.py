from collections import namedtuple
from matplotlib import pyplot as plt


Pair = namedtuple("Pair", "region1 region2 direct")


class Region:

    def __init__(self, re_range, *, buffer_region=5):
        self.start = re_range.start
        self.end = re_range.end
        self.range = buffer_region
        self.direct = re_range.direct 

    def __call__(self, value):
        if value < self.start:
            return False
        if value > self.end:
            return False
        return True

    def to_region(self, value, other):
        if not isinstance(other, Region):
            raise ValueError("other needs to be a region")
        # if value already in region, return it
        if self(value) is True:
            return value
        # if value not in other raise Exception
        if other(value) is False:
            raise ValueError("value needs to be in the 'other' region")
        direct = self.direct == other.direct
        # transform
        if direct is True:
            value = self.start + value - other.start
        else:
            value = other.end - value + self.start
        # sanity check
        if self(value) is False:
            raise ValueError("value not in region!")
        #
        return value

    def is_within_range(self, value):
        if self(value) is False:
            return False

        if value < self.start + self.range:
            return True
        if value > self.end - self.range:
            return True
        return False


class RegionRange:

    def __init__(self, start, end, direct=True):
        self.start = float(start)
        self.end = float(end)
        self.direct = direct

    def __str__(self): 
        return f"Region({self.start},{self.end})"

    def __repr__(self): 
        return f"Region({self.start},{self.end})"

    def __hash__(self):
        return hash(str(RegionRange) + f"{str(id(self))}")


class Symmetrizer:

    def __init__(self, regions, pairs):
        self.regions = regions
        self.joined_regions = pairs

    def symmetrize(self, points):
        points = self._get_regions(points)

        for regions in self.joined_regions:
            self._symmetrize(points, regions)

        return self._cleanup_points(points)

    def _cleanup_points(self, points):
        # get rid of duplicates
        points = {angle: value for values in points.values()
                  for angle, value in values}
        #
        points = [[angle, value] for angle, value in points.items()]
        # sort output
        points.sort(key=lambda x: x[0])
        return points

    def _get_regions(self, points):
        output = {region: [] for region in self.regions}
        
        for angle, value in points:
            found = False
            for region, validator in self.regions.items():
                if validator(angle) is True:
                    output[region].append([angle, value])
                    found = True
                    break
            if found is False:
                print(f"Could not find:\nangle = {angle}, value = {value}")
        return output

    def _get_smallest(self, pairs, validators):
        smallest = pairs[0]
        validator = validators[0]
        for i, (angle, value) in enumerate(pairs):
            if (value < smallest[1]):
                smallest = (angle, value)
                validator = validators[i]
        return smallest, validator

    def _symmetrize(self, points, regions):
        out = tuple(points[region] if region.direct is True else points[region][::-1]
                    for region in regions)
        zipped = zip(*out)
        #
        validators = [self.regions[region] for region in regions]
        results = []
        #
        for pairs in zipped:
            results.append(self._get_smallest(pairs, validators))
            for i, (angle, value) in enumerate(pairs):
                validator = validators[i]
                if validator.is_within_range(angle) is True:
                    results.append(([angle, value], validator))
        # set results
        for region in regions:
            r1 = self.regions[region]
            points[region] = [[r1.to_region(angle, r2), value] for (angle, value), r2 in results]





points = [[0.275, 8.40173012],
[10.275, 4.93173852],
[20.275, 1.36820037],
[30.275, 0.00047259],
[40.275, 1.47345665],
[50.275, 5.89865753],
[60.275, 13.02560255],
[70.275, 22.33722582],
[80.275, 33.1593778],
[90.275, 44.74365981],
[100.275, 56.36842703],
[110.275, 67.2381268],
[120.275, 12.89020553],
[130.275, 5.75711684],
[140.275, 1.37274249],
[150.275, 0.00456837],
[160.275, 1.61830547],
[170.275, 5.43979895],
[180.275, 8.40480195],
[190.275, 5.20444917],
[200.275, 1.45975154],
[210.275, 0.0],
[220.275, 1.53688872],
[230.275, 6.08099848],
[240.275, 13.35434136],
[250.275, 22.9201655],
[260.275, 34.18678832],
[270.275, 46.4614192],
[280.275, 58.93406527],
[290.275, 21.77972723],
[300.275, 12.57031466],
[310.275, 5.58202227],
[320.275, 1.31500775],
[330.275, 0.00446335],
[340.275, 1.51643608],
[350.275, 5.16333384]]


r1 = RegionRange(0, 90, direct=False)
r2 = RegionRange(90, 180, direct=True)
r3 = RegionRange(180, 270, direct=False)
r4 = RegionRange(270, 360, direct=True)

pairs = [[r1, r2, r3, r4]]

sym = Symmetrizer({region: Region(region) for region in (r1, r2, r3, r4)}, pairs)

points = sym.symmetrize(points)
angles = []
values = []
for angle, value in points:
    angles.append(angle)
    values.append(value)
#    print(angle, value)

plt.plot(angles, values)
plt.show()
