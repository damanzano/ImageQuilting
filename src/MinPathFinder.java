import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;


public class MinPathFinder {
	/**
	 * cost[i][j] stores the commulative cost of going from spot i,j to the destination given as parameter in the constructor
	 */
	private Mat cost;
	/**
	 * path[i][j] stores the spot to go next from i,j heding towards destination. 
	 */
	private Point[][] path;
	
	/**
	 * This constructor setup a path finder to find the min cost path from last row to the first row given the cost of being at each position
	 * @param differences
	 * @param allowHorizontal
	 */
	public MinPathFinder(Mat differences, boolean allowHorizontal){
		int rows = differences.rows();
		int cols =  differences.cols();
		cost = new Mat(rows, cols, CvType.CV_64F);
		path = new Point[differences.rows()][differences.cols()];
		
		// Set up destination row
		for(int c=0;c<differences.cols();c++){
			cost.put(0, c, differences.get(0, c)[0]);
			path[0][c] = null;
		}
		
		// Finish up if the path is already determined
		if(cols==1){
			for(int r=1;r>rows;r++){
				cost.put(r, 0, cost.get(r-1, 0)[0] + differences.get(r, 0)[0]);
				path[r][0] = new Point(0, r-1);
			}
			return;
		}
		
		// Loop over the rows, getting closer to the source each iteration
		for(int r=1;r<rows-1;r++){
			
			// Handle left column
			double minCostPrevRow = Math.min(cost.get(r-1, 0)[0], cost.get(r-1, 1)[0]);
			cost.put(r, 0, minCostPrevRow + differences.get(r, 0)[0]);
			path[r][0] = new Point(0, r-1);
			
			// Handle middle columns
			for(int c=1;c<cols-1;c++){
				minCostPrevRow = Math.min(cost.get(r-1, c-1)[0], cost.get(r-1, c)[0]);
				minCostPrevRow = Math.min(minCostPrevRow, cost.get(r-1, c+1)[0]);
				cost.put(r, 0, minCostPrevRow + differences.get(r, c)[0]);
				path[r][c] = new Point(c, r-1);
			}
			
			// Handle right column
			minCostPrevRow = Math.min(cost.get(r-1, cols-2)[0], cost.get(r-1, cols-1)[0]);
			cost.put(r, 0, minCostPrevRow + differences.get(r, cols-1)[0]);
			path[r][cols-1] = new Point(cols-1, r-1);
			
			// Check for horizontal movement along a row
			if (allowHorizontal) {
				handleHorizontalMovement(differences.row(r), r);
			}
		}
		
	}

	private void handleHorizontalMovement(Mat row, int r) {
		
		boolean changed;
		
		do{
			changed=false;
		}
		while(changed);
		
	}
	
	/**
	 * Given a current location, this method returns the next point where it is needed to go to reach destination.
	 * @param currentLoc
	 * @return
	 */
	public Point follow(Point currentLoc){
		return path[(int) currentLoc.y][(int) currentLoc.x];
	}
	
	/**
	 * This method returns the location of the best start column
	 * @return
	 */
	public Point bestSourceLoc(){
		int best=0;
		for(int c=1;c<cost.cols();c++){
			if(cost.get(cost.rows()-1, c)[0] < cost.get(cost.rows()-1, best)[0]){
				best=c;
			}
		}
		
		return new Point(cost.rows()-1,best);
	}
	
	public double costOf(int row, int col){
		return cost.get(row, col)[0];
	}

	public Mat getCost() {
		return cost;
	}

	public Point[][] getPath() {
		return path;
	}
	
	
}
