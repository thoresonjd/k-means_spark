"""Justin Thoreson
22 Feb 2022
k_means.py

This script implements the K-means clustering algorithm in Spark
"""

from pyspark import SparkContext
import sys

def squaredDist(p1 : tuple, p2 : tuple) -> float:
    """Computes the squared distance between two points

    :param p1: The first point
    :param p2: The second point
    :return: The squared distance between p1 and p2
    """

    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def getClosestCenter(point : tuple, centers : list) -> tuple:
    """Calculates the closest center from a point
    
    :param point: A data point 
    :param centers: A list containing center points
    :return: A tuple containing the closest center as key and the point as value
    """

    closestCenter = None
    shortestDist = float('inf')
    for center in centers:
        if (dist := squaredDist(point, center)) < shortestDist:
            shortestDist = dist
            closestCenter = center
    return (closestCenter, point)

def computeNewCenter(points: list) -> tuple:
    """Computes the new centerpoint for each cluster
    
    :param cluster: A cluster of points
    :return: The new center of the cluster
    """

    latSum = lonSum = 0
    numPoints = len(points)
    for lat, lon in points:
        latSum += lat
        lonSum += lon
    return (latSum/numPoints, lonSum/numPoints)

def k_means(sc : SparkContext, k : int, convergeDist : float) -> None:
    """Runs the K-means clustering algorithm
    
    :param sc: A SparkContext object
    :param k: The number of center points
    :param convergeDist: The minimum convergence distance
    """

    # Read and clean data
    points = sc \
        .textFile(sys.argv[1]) \
        .map(lambda line : line.split(',')) \
        .map(lambda line : (float(line[-2]), float(line[-1]))) \
        .filter(lambda coords : coords != (0, 0)) \
        .persist()

    # 1. Choose k random points as starting centers
    centers = points.takeSample(False, k, 34)

    tempDist = float('inf')
    while tempDist > convergeDist:

        # 2. Find all points closest to each center
        clusters = points \
            .map(lambda point : getClosestCenter(point, centers)) \
            .groupByKey()

        # 3. Find new centerpoint of each cluster
        newCenters = clusters \
            .map(lambda cluster : (cluster[0], computeNewCenter(list(cluster[1])))) \
            .persist()
        
        centers = newCenters \
            .map(lambda centers : centers[1]) \
            .collect()

        # 4. Compare total squared distance to convergence distance
        tempDist = newCenters \
            .map(lambda center : squaredDist(center[0], center[1])) \
            .reduce(lambda v1, v2: v1 + v2)

    # Output final centers
    sc.parallelize(centers).saveAsTextFile(sys.argv[2])

def main() -> None:
    """Implements the K-means clustering algorithm"""

    if len(sys.argv) != 3:
        print('Usage: k_means.py <input> <output>', file=sys.stderr)
        exit(-1)

    sc = SparkContext()
    k = 5
    convergeDist = 0.1
    k_means(sc, k, convergeDist)
    sc.stop()

if __name__ == '__main__':
    main()
