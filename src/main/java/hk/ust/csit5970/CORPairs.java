package hk.ust.csit5970;

import org.apache.commons.cli.*;
import org.apache.commons.cli.Options;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Compute the bigram correlation coefficients using "pairs" approach.
 */
public class CORPairs extends Configured implements Tool {
	private static final Logger LOG = Logger.getLogger(CORPairs.class);

	/*
	 * First-pass Mapper: 统计每个单词的总频数 (Freq(A))
	 * 每行中对每个单词出现都输出一次计数
	 */
	private static class CORMapper1 extends Mapper<LongWritable, Text, Text, IntWritable> {
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			// 使用指定的 tokenizer，对文本进行预处理
			String clean_doc = value.toString().replaceAll("[^a-z A-Z]", " ");
			StringTokenizer doc_tokenizer = new StringTokenizer(clean_doc);
			// 对每个单词输出 (word,1)
			while (doc_tokenizer.hasMoreTokens()) {
				String word = doc_tokenizer.nextToken().toLowerCase();
				context.write(new Text(word), new IntWritable(1));
			}
		}
	}

	/*
	 * First-pass Reducer: 汇总每个单词的频数，输出到中间结果文件 mid/part-r-00000
	 */
	private static class CORReducer1 extends Reducer<Text, IntWritable, Text, IntWritable> {
		@Override
		public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			context.write(key, new IntWritable(sum));
		}
	}


	/*
	 * Second-pass Mapper: 对每行数据，统计该行中不重复的单词组合 (A, B)
	 * 注意：每行只计数一次，且输出的单词对按字典序排列（即 A < B）
	 */
	public static class CORPairsMapper2 extends Mapper<LongWritable, Text, PairOfStrings, IntWritable> {
		@Override
		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			// 使用 tokenizer 预处理文本
			String clean_doc = value.toString().replaceAll("[^a-z A-Z]", " ");
			StringTokenizer doc_tokenizer = new StringTokenizer(clean_doc);
			// 使用 Set 去除重复单词
			Set<String> uniqueWords = new HashSet<>();
			while (doc_tokenizer.hasMoreTokens()) {
				uniqueWords.add(doc_tokenizer.nextToken().toLowerCase());
			}
			// 将单词转换为列表并排序，保证组合中顺序固定（即字典序较小的单词为 left）
			List<String> wordList = new ArrayList<>(uniqueWords);
			Collections.sort(wordList);
			// 构造所有不重复的单词对 (A, B) (A < B)
			for (int i = 0; i < wordList.size(); i++) {
				for (int j = i + 1; j < wordList.size(); j++) {
					PairOfStrings pair = new PairOfStrings(wordList.get(i), wordList.get(j));
					context.write(pair, new IntWritable(1));
				}
			}
		}
	}

	/*
	 * Second-pass Combiner: 在 Mapper 端对相同单词对的计数进行局部聚合
	 */
	private static class CORPairsCombiner2 extends Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
		@Override
		protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			context.write(key, new IntWritable(sum));
		}
	}

	/*
	 * Second-pass Reducer: 汇总每个单词对的出现行数 Freq(A, B)
	 * 然后根据预先加载的中间结果 word_total_map，计算相关系数：
	 *    COR(A, B) = Freq(A, B) / (Freq(A) * Freq(B))
	 */
	public static class CORPairsReducer2 extends Reducer<PairOfStrings, IntWritable, PairOfStrings, DoubleWritable> {
		private final static Map<String, Integer> word_total_map = new HashMap<String, Integer>();

		/*
		 * Preload the middle result file.
		 * 中间结果文件中，每行格式为: word \t frequency
		 */
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			Path middle_result_path = new Path("mid/part-r-00000");
			Configuration middle_conf = new Configuration();
			try {
				FileSystem fs = FileSystem.get(URI.create(middle_result_path.toString()), middle_conf);
				if (!fs.exists(middle_result_path)) {
					throw new IOException(middle_result_path.toString() + " not exist!");
				}
				FSDataInputStream in = fs.open(middle_result_path);
				InputStreamReader inStream = new InputStreamReader(in);
				BufferedReader reader = new BufferedReader(inStream);
				LOG.info("Reading middle result...");
				String line = reader.readLine();
				String[] line_terms;
				while (line != null) {
					line_terms = line.split("\t");
					if(line_terms.length >= 2){
						word_total_map.put(line_terms[0], Integer.valueOf(line_terms[1]));
					}
					line = reader.readLine();
				}
				reader.close();
				LOG.info("Finished reading middle result.");
			} catch (Exception e) {
				System.out.println(e.getMessage());
			}
		}

		@Override
		protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int pairCount = 0;
			for (IntWritable val : values) {
				pairCount += val.get();
			}
			// 从预加载的 word_total_map 中获取单词 A 和 B 的频数
			Integer freqA = word_total_map.get(key.getLeftElement());
			Integer freqB = word_total_map.get(key.getRightElement());
			if (freqA == null || freqB == null || freqA == 0 || freqB == 0) {
				// 如果没有找到相应频数，则跳过该对
				return;
			}
			double corr = pairCount / (freqA.doubleValue() * freqB.doubleValue());
			context.write(key, new DoubleWritable(corr));
		}
	}

	private static final class MyPartitioner extends Partitioner<PairOfStrings, FloatWritable> {
		@Override
		public int getPartition(PairOfStrings key, FloatWritable value, int numReduceTasks) {
			return (key.getLeftElement().hashCode() & Integer.MAX_VALUE) % numReduceTasks;
		}
	}

	/**
	 * Creates an instance of this tool.
	 */
	public CORPairs() {
	}

	private static final String INPUT = "input";
	private static final String MIDDLE = "middle";
	private static final String OUTPUT = "output";
	private static final String NUM_REDUCERS = "numReducers";

	/**
	 * Runs this tool.
	 */
	@SuppressWarnings({ "static-access" })
	public int run(String[] args) throws Exception {
		Options options = new Options();

		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("input path").create(INPUT));
		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("output path").create(OUTPUT));
		options.addOption(OptionBuilder.withArgName("num").hasArg()
				.withDescription("number of reducers").create(NUM_REDUCERS));

		CommandLine cmdline;
		CommandLineParser parser = new GnuParser();

		try {
			cmdline = parser.parse(options, args);
		} catch (ParseException exp) {
			System.err.println("Error parsing command line: " + exp.getMessage());
			return -1;
		}

		// Lack of arguments
		if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
			System.out.println("args: " + Arrays.toString(args));
			HelpFormatter formatter = new HelpFormatter();
			formatter.setWidth(120);
			formatter.printHelp(this.getClass().getName(), options);
			ToolRunner.printGenericCommandUsage(System.out);
			return -1;
		}

		String inputPath = cmdline.getOptionValue(INPUT);
		String middlePath = "mid";
		String outputPath = cmdline.getOptionValue(OUTPUT);

		int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

		LOG.info("Tool: " + CORPairs.class.getSimpleName());
		LOG.info(" - input path: " + inputPath);
		LOG.info(" - output path: " + outputPath);
		LOG.info(" - number of reducers: " + reduceTasks);

		// Setup for the first-pass MapReduce (计算单词频数)
		Configuration conf1 = new Configuration();

		Job job1 = Job.getInstance(conf1, "Firstpass");

		job1.setJarByClass(CORPairs.class);
		job1.setMapperClass(CORMapper1.class);
		job1.setReducerClass(CORReducer1.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(IntWritable.class);

		FileInputFormat.setInputPaths(job1, new Path(inputPath));
		FileOutputFormat.setOutputPath(job1, new Path(middlePath));

		// Delete the output directory if it exists already.
		Path middleDir = new Path(middlePath);
		FileSystem.get(conf1).delete(middleDir, true);

		// Time the program
		long startTime = System.currentTimeMillis();
		job1.waitForCompletion(true);
		LOG.info("Job 1 Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

		// Setup for the second-pass MapReduce (计算 bigram 的相关系数)

		// Delete the output directory if it exists already.
		Path outputDir = new Path(outputPath);
		FileSystem.get(conf1).delete(outputDir, true);

		Configuration conf2 = new Configuration();
		Job job2 = Job.getInstance(conf2, "Secondpass");

		job2.setJarByClass(CORPairs.class);
		job2.setMapperClass(CORPairsMapper2.class);
		job2.setCombinerClass(CORPairsCombiner2.class);
		job2.setReducerClass(CORPairsReducer2.class);

		job2.setOutputKeyClass(PairOfStrings.class);
		job2.setOutputValueClass(DoubleWritable.class);
		job2.setMapOutputValueClass(IntWritable.class);
		job2.setNumReduceTasks(reduceTasks);

		FileInputFormat.setInputPaths(job2, new Path(inputPath));
		FileOutputFormat.setOutputPath(job2, new Path(outputPath));

		// Time the program
		startTime = System.currentTimeMillis();
		job2.waitForCompletion(true);
		LOG.info("Job 2 Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

		return 0;
	}

	/**
	 * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
	 */
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new CORPairs(), args);
	}
}

