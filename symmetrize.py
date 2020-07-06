from collections import namedtuple
from matplotlib import pyplot as plt


Pair = namedtuple("Pair", "region1 region2 direct")


class Region:

    def __init__(self, re_range, *, buffer_region=5):
        self.start = re_range.start
        self.end = re_range.end
        self.range = buffer_region

    def __call__(self, value):
        if value < self.start:
            return False
        if value > self.end:
            return False
        return True

    def to_region(self, value, other, *, direct=False):
        if not isinstance(other, Region):
            raise ValueError("other needs to be a region")
        # if value already in region, return it
        if self(value) is True:
            return value
        # if value not in other raise Exception
        if other(value) is False:
            raise ValueError("value needs to be in the 'other' region")
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

    def __init__(self, start, end):
        self.start = float(start)
        self.end = float(end)

    def __str__(self): 
        return f"Region({self.start},{self.end})"

    def __repr__(self): 
        return f"Region({self.start},{self.end})"

    def __hash__(self):
        return hash(str(RegionRange) + f"{str(id(self))}")


class Symmetrizer:

    def __init__(self, regions, pairs):
        self.regions = regions
        self.pairs = pairs

    def symmetrize(self, points):
        points = self._get_regions(points)

        for (region1, region2, direct) in self.pairs:
            self._symmetrize_pair(points, region1, region2, direct=direct)

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

    def _symmetrize_pair(self, points, region1, region2, *, direct=False):
        if direct is True:
            zipped = zip(points[region1], points[region2])
        else:
            zipped = zip(points[region1], points[region2][::-1])
        #
        validator_1 = self.regions[region1]
        validator_2 = self.regions[region2]
        #
        results = []
        #
        for (angle1, value1), (angle2, value2) in zipped:
            if value1 <= value2:
                results.append([angle1, value1])
            else:
                results.append([angle2, value2])
            #
            if validator_1.is_within_range(angle1) is True:
                results.append([angle1, value1])
            #
            if validator_2.is_within_range(angle2) is True:
                results.append([angle2, value2])
        
        # set results
        points[region1] = [[validator_1.to_region(angle, validator_2, direct=direct), value] 
                           for angle, value in results]
        points[region2] = [[validator_2.to_region(angle, validator_1, direct=direct), value] 
                           for angle, value in results]





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


r1 = RegionRange(0, 90)
r2 = RegionRange(90, 180)
r3 = RegionRange(180, 270)
r4 = RegionRange(270, 360)

pairs = [Pair(r1, r2, False), 
         Pair(r1, r3, True),
         Pair(r1, r4, False),
         Pair(r2, r3, False),
         Pair(r2, r4, True),
         Pair(r3, r4, False)]

sym = Symmetrizer({region: Region(region) for region in (r1, r2, r3, r4)}, pairs)

points = sym.symmetrize(points)
angles = []
values = []
for angle, value in points:
    angles.append(angle)
    values.append(value)
    print(angle, value)

plt.plot(angles, values)
plt.show()
