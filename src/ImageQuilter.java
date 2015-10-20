import java.io.File;
import java.util.logging.Logger;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;

public class ImageQuilter {
	private Mat textureImage;
	private int patchSize;
	private int overlapSize;
	private boolean allowHorizontalPaths;
	private double pathCostWeight;

	public static int DEFAULT_PATCH_SIZE = 36;
	public static int DEFAULT_OVERLAP_SIZE = 6;

	/**
	 * Load the OpenCV system library
	 */
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	/**
	 * Sets up the algorithm.
	 * 
	 * @param textureImage
	 *            This is the texture to sample from.
	 * @param patchSize
	 *            This is the width (pixels) of the square patches used.
	 * @param overlapSize
	 *            This is the width (pixels) of the overlap region.
	 * @param allowHorizontalPaths
	 *            When finding min paths, can the path travel along a stage?
	 * @param pathCostWeight
	 *            The SSD for the overlap region and the min SSD path cost have
	 *            the same range. The total cost is then pathCost*pathCostWeight
	 *            plus ssd*(1-pathCostWeight).
	 */
	public ImageQuilter(Mat textureImage, int patchsize, int overlapsize,
			boolean allowHorizontalPaths, double pathCostWeight) {
		this.textureImage = textureImage;
		this.overlapSize = overlapsize;
		this.patchSize = patchsize;

		this.allowHorizontalPaths = allowHorizontalPaths;
		this.pathCostWeight = pathCostWeight;
	}

	/**
	 * This method synthesizes a new texture image with the given dimensions.
	 */
	public Mat synthesize(int outWidth, int outHeight) {

		if (outWidth < patchSize || outHeight < patchSize) {
			throw new IllegalArgumentException("Output size is too small");
		}

		// Calculate the optimal width and height
		int patchCols = Math.round((float) (outWidth - patchSize)
				/ (patchSize - overlapSize));
		int patchRows = Math.round((float) (outHeight - patchSize)
				/ (patchSize - overlapSize));
		int okWidth = patchCols * (patchSize - overlapSize) + patchSize;
		int okHeight = patchRows * (patchSize - overlapSize) + patchSize;
		patchCols++;
		patchRows++;

		// Check if the output size is acceptable and fix it if it is not the
		// case
		if (okWidth != outWidth || okHeight != outHeight) {
			System.out.println("Your output size requires partial"
					+ " patches that are currently" + " not supported.");
			outWidth = okWidth;
			outHeight = okHeight;
			System.out.println("Using width = " + outWidth + " and  height = "
					+ outHeight + " instead.");
		}

		// Create the output image
		Mat output = new Mat(okHeight, okWidth, CvType.CV_8UC3);

		// Choose a random place to get the first patch
		int x = (int) (Math.random() * (textureImage.cols() - patchSize));
		int y = (int) (Math.random() * (textureImage.rows() - patchSize));

		// Save the best set of distances between patches
		double dists[][] = new double[textureImage.cols() - patchSize][textureImage
				.rows() - patchSize];

		for (int r = 0; r < okWidth; r += patchSize - overlapSize) {
			for (int c = 0; c < okHeight; c += patchSize - overlapSize) {
				Mat outputCell = output.submat(new Rect(c, r, patchSize,
						patchSize));

				// Get a crop of the source texture image
				Rect roi = new Rect(x, y, patchSize, patchSize);
				Mat cropped = new Mat(textureImage, roi);

				// Replace the output cell with the selected source patch
				cropped.copyTo(outputCell);

				// Get the patches of source texture image that satisfy the
				// overlap constraints
				Point bestLoc = calcDists(dists, outputCell, c, r);
				double bestval = dists[(int)bestLoc.x][(int)bestLoc.y];
			}
		}

		// Write the image on disk
		Imgcodecs.imwrite("output/step.jpg", output);

		// // View inView = new View(input, x, y);
		// // Patch outPatch = new Patch(output, 0, 0, patchsize, patchsize);
		// // SynthAide.copy(inView, outPatch, 0, 0, patchsize, patchsize);
		//
		// // done already?
		// if (!outPatch.nextColumn(overlapsize))
		// return output;
		//
		// // loop over the rows of output patches
		// int currow = 0;
		// double dists[][] = new double[input.getHeight() - patchsize +
		// 1][input
		// .getWidth() - patchsize + 1];
		// do {
		//
		// // loop over the patches in this row
		// do {
		//
		// // get the distances for this neighborhood
		// TwoDLoc bestloc = calcDists(dists, outPatch);
		// double bestval = dists[bestloc.getRow()][bestloc.getCol()];
		//
		// // pick one of the close matches
		// double threshold = bestval * 1.1;
		// LinkedList loclist = SynthAide.lessThanEqual(dists, threshold);
		// int choice = (int) (Math.random() * loclist.size());
		// TwoDLoc loc = (TwoDLoc) loclist.get(choice);
		//
		// // copy in the patch
		// // fillAndBlend(outPatch, loc);
		// pathAndFill(outPatch, loc);
		//
		// } while (outPatch.nextColumn(overlapsize));
		//
		// currow++;
		// System.out.println("done with row " + currow + " / " + patchRows);
		//
		// } while (outPatch.nextRow(overlapsize));

		return output;
	}

	/**
	 * This method calculates the distance (SSD) between the overlap part of
	 * outPatch and the corresponding parts of the possible input patches. If
	 * the pathCostWeight extension has been activated, this will also calculate
	 * the path cost and weight the distance based on that cost. This returns
	 * the array index of the smallest distance found.
	 * 
	 * @param dists
	 *            This will be filled in. The return value in dists[y][x] will
	 *            be the SSD between an input patch with corner (x,y) and the
	 *            given output patch.
	 */
	private Point calcDists(double[][] dists, Mat outputCell, int cellCol,
			int cellRow) {

		double best = Double.MAX_VALUE;
		Point bestloc = null;

		// loop over the possible input patch row locations
		for (int y = 0; y < textureImage.rows(); y += patchSize) {
			for (int x = 0; x > textureImage.cols(); x += patchSize) {
				Mat sourceCell = textureImage.submat(new Rect(x, y, patchSize,
						patchSize));

				double sum = 0.0;
				int count = 0;
				Mat leftOverlapDiff = null;
				Mat topOverlapDiff = null;

				// Calculate ssd of left overlap
				if (cellCol != 0) {
					Mat sourceLeft = sourceCell.submat(new Rect(0, 0,
							patchSize, sourceCell.rows()));
					Mat outputRight = outputCell.submat(new Rect(outputCell
							.cols() - patchSize, 0, patchSize, outputCell
							.rows()));
					Core.subtract(outputRight, sourceLeft, leftOverlapDiff);
					Core.pow(leftOverlapDiff, 2, leftOverlapDiff);
					sum += Core.sumElems(leftOverlapDiff).val[0];
				}

				// Calculate ssd of top overlap
				if (cellRow != 0) {
					Mat sourceTop = sourceCell.submat(new Rect(0, 0, patchSize,
							sourceCell.cols()));
					Mat outputBottom = outputCell.submat(new Rect(0, outputCell
							.rows() - patchSize, outputCell.cols(), patchSize));
					Core.subtract(outputBottom, sourceTop, topOverlapDiff);
					Core.pow(topOverlapDiff, 2, topOverlapDiff);
					sum += Core.sumElems(topOverlapDiff).val[0];
				}

				// save the total and compare to the best yet
				dists[y][x] = sum;
				if (sum < best) {
					best = sum;
					bestloc = new Point(x, y);
				}
			}
		}

//		// do we weight the SSD with the min cost path cost?
//		if (pathCostWeight > 0) {
//
//			double cost = avgCostOfBestPath(leftoverlap, topoverlap);
//
//			// update the sum appropriately
//			cost = cost / (255 * 255);
//			sum = sum * (1 - pathCostWeight) + pathCostWeight * cost;
//		}

		return bestloc;
	}

	public static void main(String[] args) {
		File textureFile = new File("resources/textures/0.jpg");
		Mat textureImage = new Mat();

		// read the images
		textureImage = Imgcodecs.imread(textureFile.getAbsolutePath(),
				Imgcodecs.CV_LOAD_IMAGE_COLOR);

		ImageQuilter iq = new ImageQuilter(textureImage, 30, 5, false, 2.5);
		iq.synthesize(150, 150);

	}

}
