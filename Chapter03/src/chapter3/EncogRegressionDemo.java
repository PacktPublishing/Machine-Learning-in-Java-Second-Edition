package chapter3;

import java.io.File;

import org.encog.ConsoleStatusReportable;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.model.EncogModel;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

public class EncogRegressionDemo {
	public static void main(String args[]) {
		try
		{
			File datafile = new File("data/ENB2012_data.csv");
			VersatileDataSource source = new CSVDataSource(datafile, true, CSVFormat.DECIMAL_POINT);
			VersatileMLDataSet data = new VersatileMLDataSet(source); 
			data.defineSourceColumn("X1", 0, ColumnType.continuous); 
			data.defineSourceColumn("X2", 1, ColumnType.continuous); 
			data.defineSourceColumn("X3", 2, ColumnType.continuous); 
			data.defineSourceColumn("X4", 3, ColumnType.continuous);
			data.defineSourceColumn("X5", 4, ColumnType.continuous);
			data.defineSourceColumn("X6", 5, ColumnType.continuous);
			data.defineSourceColumn("X7", 6, ColumnType.continuous);
			data.defineSourceColumn("X8", 7, ColumnType.continuous);
			
			ColumnDefinition outputColumn1 = data.defineSourceColumn("Y1", 8, 
				     ColumnType.continuous);
			ColumnDefinition outputColumn2 = data.defineSourceColumn("Y2", 9, 
				     ColumnType.continuous);
			
			ColumnDefinition outputscol [] = {outputColumn1, outputColumn2};
			data.analyze();
			
			data.defineMultipleOutputsOthersInput(outputscol);
			
			EncogModel model = new EncogModel(data); 
			model.selectMethod(data, MLMethodFactory.TYPE_FEEDFORWARD);
			model.setReport(new ConsoleStatusReportable());
			
			data.normalize();
			
			model.holdBackValidation(0.3, true, 1001);
			
			model.selectTrainingType(data);
			
			MLRegression bestMethod = (MLRegression)model.crossvalidate(5, true);
			
			NormalizationHelper helper = data.getNormHelper(); 
			
			System.out.println(helper.toString()); 
		    
			
			System.out.println("Final model: " + bestMethod); 
			
			
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
}
