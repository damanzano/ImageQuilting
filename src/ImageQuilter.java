import java.io.File;
import java.util.LinkedList;
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
		Mat output = Mat.zeros(okHeight, okWidth, CvType.CV_8UC3);
		
		// Get the first patch to start the process
		selectFirstRandomPatch(output);
		Imgcodecs.imwrite("output/process-1.jpg", output);

		// Save the best set of distances between patches
		double dists[][] = new double[textureImage
				.rows() - patchSize][textureImage.cols() - patchSize];
		//
		for (int r = 0; (r + patchSize - overlapSize) < okWidth; r += patchSize - overlapSize) {
			for (int c = 0; (c + patchSize - overlapSize) < okHeight; c += patchSize - overlapSize) {
				// Get the output cell to be analyzed
				Point outputLoc = new Point(c,r);
				Mat outputCell = output.submat(new Rect(c, r, patchSize,
						patchSize));

				// Get the all patches of source texture image and their differences 
				Point bestLoc = calcDists(dists, outputCell, c, r);
				double bestval = dists[(int)bestLoc.y][(int)bestLoc.x];
				
				
				// Filter the ones that satisfy the overlap constraints
				double threshold = bestval*1.1;
				LinkedList<Point> loclist = getBestOverlaps(dists, threshold);
				int choice = (int) (Math.random() * loclist.size());
				Point loc = loclist.get(choice);
				
				// Fill the output with new data
				fillPatch(outputCell, outputLoc, loc);
			}
		}

		// Write the image on disk
		Imgcodecs.imwrite("output/step-2.jpg", output);
		return output;
	}
	
	

	/**
	 * This method selects a random patch from the source texture and put it in
	 * top left corner of an output image
	 * 
	 * @param output
	 *            The output image to be synthesized
	 */
	private void selectFirstRandomPatch(Mat output) {
		// Choose a random place to get the first patch
		int x = (int) (Math.random() * (textureImage.cols() - patchSize));
		int y = (int) (Math.random() * (textureImage.rows() - patchSize));

		// Get a crop of the source texture image
		Rect roi = new Rect(x, y, patchSize, patchSize);
		Mat cropped = new Mat(textureImage, roi);

		Mat firstCell = output.submat(new Rect(0, 0, patchSize, patchSize));

		// Replace the output cell with the selected source patch
		cropped.copyTo(firstCell);
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
	 * @param outputCell
	 * 			The output cell to be analyzed
	 * @param cellCol
	 * 			Current column of output cell on the whole output
	 * @param cellRow
	 * 			Current row of output cell on the whole output
	 */
	private Point calcDists(double[][] dists, Mat outputCell, int cellCol,
			int cellRow) {

		double best = Double.MAX_VALUE;
		Point bestloc = null;
		

		// loop over the possible input patch row locations
		for (int y = 0; y < textureImage.rows() - patchSize; y++) {
			for (int x = 0; x < textureImage.cols() - patchSize; x++) {
				Mat sourceCell = textureImage.submat(new Rect(x, y, patchSize,
						patchSize));

				double sum = 0.0;
				Mat leftOverlapDiff = new Mat();
				Mat topOverlapDiff = new Mat();

				// Calculate ssd of left overlap
				if (cellCol != 0) {
					Mat sourceLeft = sourceCell.submat(new Rect(0, 0, overlapSize, sourceCell.rows()));
					Mat outputLeft = outputCell.submat(new Rect(0, 0, overlapSize, outputCell.rows()));
					Core.subtract(outputLeft, sourceLeft, leftOverlapDiff);
					Core.pow(leftOverlapDiff, 2, leftOverlapDiff);
					sum += Core.sumElems(leftOverlapDiff).val[0];
				}

				// Calculate ssd of top overlap
				if (cellRow != 0) {
					Mat sourceTop = sourceCell.submat(new Rect(0, 0, sourceCell.cols(), overlapSize));
					Mat outputTop = outputCell.submat(new Rect(0, 0, outputCell.cols(), overlapSize));
					Core.subtract(outputTop, sourceTop, topOverlapDiff);
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

	
	/**
	 * This method return a list of the top left points of overlaps that satisfy
	 * the threshold difference
	 * 
	 * @param vals
	 * @param threshold
	 * @return
	 */
	private LinkedList<Point> getBestOverlaps(double[][] vals, double threshold){
		LinkedList<Point> list = new LinkedList<>();
		for(int r=0; r<vals.length; r++){
			for(int c=0; c<vals[r].length; c++){
				if(vals[r][c] >= 0 && vals[r][c] <=threshold){
					list.addFirst(new Point(c, r));
				}
			}
		}
		return list;
	}
	
	/**
	 * 
	 * @param outputCell
	 * @param loc
	 */
	private void fillPatch(Mat outputCell, Point outputLoc, Point sourceLoc) {
		Mat sourceCell = textureImage.submat(new Rect((int) sourceLoc.x, (int) sourceLoc.y, patchSize, patchSize));
		
		
		if(outputLoc.x==0){
			
		}
		
		int nonOverlapSize = patchSize-overlapSize;
		Mat sourceNOCell = sourceCell.submat(new Rect(overlapSize,overlapSize, nonOverlapSize, nonOverlapSize));
		Mat outputNOCell = sourceCell.submat(new Rect(overlapSize,overlapSize, nonOverlapSize, nonOverlapSize));
		sourceCell.copyTo(outputCell);
		//sourceNOCell.copyTo(outputNOCell);
		
	}
	
	public static void main(String[] args) {
		File textureFile = new File("resources/textures/0.jpg");
		Mat textureImage = new Mat();

		// read the images
		textureImage = Imgcodecs.imread(textureFile.getAbsolutePath(),
				Imgcodecs.CV_LOAD_IMAGE_COLOR);

		ImageQuilter iq = new ImageQuilter(textureImage, 30, 5, false, 2.5);
		iq.synthesize(155, 155);

	}

}
