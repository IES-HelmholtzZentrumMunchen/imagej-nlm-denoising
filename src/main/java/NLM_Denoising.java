import Jama.Matrix;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.Calibration;
import ij.plugin.filter.PlugInFilter;
import ij.process.ImageProcessor;
import ij.gui.*;

//import java.awt.*;
import java.awt.*;
import java.awt.geom.Arc2D;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * NLM Denoising algorithm
 *
 * A plugin for ImageJ 1.x to denoise images using the non-local means (NLM) algorithm.
 * The noise is assumed to be the result of an additive Gaussian process or the result of a Poisson/Gaussian process.
 *
 * @author Julien Pontabry
 */
public class NLM_Denoising implements PlugInFilter/*, DialogListener*/ {
    /**
     * Input stack image.
     */
    private ImagePlus m_image = null;

    /**
     * Size of the patch (default: 2*2+1 size).
     */
    private int m_patchSize = 5;

    /**
     * Size of the search window (default: 2*5+1 size).
     */
    private int m_windowSize = 11;

    /**
     * Variance of the image (in case of Gaussian noise) or zero for automatic detection.
     */
    private double m_variance = 0;

    /**
     * Decay control between 0 and 1. It should be close to 1 for Gaussian distribution.
     */
    private double m_decayControl = 0.3;

    /**
     * Blockwise reduction factor (step size)
     */
    private int m_blockReductionFactor = 2;

    /**
     * Try to keep more the details (more noisy).
     */
    private boolean m_keepDetails = true;

    /**
     * True if 3D denoising, false for 2D/stack denoising.
     */
    private boolean m_3D = false;

    /**
     * Low light conditions option (Poisson/Gaussian statistics).
     */
    private boolean m_lowLight = false;

    /**
     * Progress information (Java does not support mutable double natively).
     */
    private int m_currentProgress, m_maxProgress;

    /**
     * @see ij.plugin.filter.PlugInFilter#setup(java.lang.String, ij.ImagePlus)
     */
    @Override
    public int setup(String arg, ImagePlus imp) {
        // Check if there is an image
        if(imp == null) {
            IJ.error("NLM Denoising", "An image is required for this plugin !");
            return DONE;
        }

        if(imp.getNChannels() > 1) {
            IJ.error("NLM Denoising", "Multiple channels are not supported !");
            return DONE;
        }

        // Display GUI
        GenericDialog gui = new GenericDialog("NLM Denoising");
        gui.addPanel(new SeparatorPanel("Smoothness control"));
        gui.addNumericField("Smoothness decay", m_decayControl, 1);
        gui.addCheckbox("Keep details", m_keepDetails);

        gui.addPanel(new SeparatorPanel("Speedness control"));
        gui.addNumericField("Reduction factor", m_blockReductionFactor, 0);

        gui.addPanel(new SeparatorPanel("Neighborhood options"));
        gui.addNumericField("Patch size", m_patchSize, 0);
        gui.addNumericField("Search window size", m_windowSize, 0);

        gui.addPanel(new SeparatorPanel("Other options"));
        gui.addNumericField("Noise variance", m_variance, 1);
        gui.addCheckbox("True_3D", m_3D);
        gui.addCheckbox("Low light", m_lowLight);

        // NOTE: This implies that the arguments cannot be recorded adn/or passed to plugin
//        gui.addDialogListener(this);    // The parameters are live-checked in the event listener
        gui.showDialog();

        // Checking GUI events
        if(gui.wasCanceled()) {
            return DONE;
        }


        if(gui.invalidNumber())
        {
            IJ.error("NLM Denoising", "A numerical field is not filled correctly !");
            return DONE;
        }

        // Get back the parameters from the GUI
        m_decayControl         = gui.getNextNumber();       // Decay control
        m_keepDetails          = gui.getNextBoolean();      // Try to keep details (more noisy)
        m_blockReductionFactor = (int)gui.getNextNumber();  // Reduction factor of the blockwise NLM
        m_patchSize            = (int)gui.getNextNumber();  // Patch size parameter
        m_windowSize           = (int)gui.getNextNumber();  // Search window size parameter
        m_variance             = gui.getNextNumber();       // Variance of noise (0 for automatic estimation)
        m_3D                   = gui.getNextBoolean();      // 3D or 2D/stack denoising
        m_lowLight             = gui.getNextBoolean();      // Low light conditions option

        // Check parameters
        if(m_patchSize < 3 ||
                m_windowSize <= m_patchSize ||
                Double.compare(m_variance,0.0) < 0 ||
                (m_patchSize-1 <= m_blockReductionFactor)
                ) {
            IJ.error("NLM Denoising", "The parameters are not correct ! Should be: \nwindow size > patch size\nvariance >= 0\nreduction factor < patch size - 1");
            return DONE;
        }

        // Ensure that the sizes are odd numbers (ie. of the form 2*n+1)
        if(m_patchSize % 2 == 0) {
            m_patchSize = m_patchSize + 1;
        }

        if(m_windowSize % 2 == 0) {
            m_windowSize = m_windowSize + 1;
        }

        // Get image
        m_image = imp;

        // Return process
        return DOES_8G+DOES_16+NO_CHANGES;
    }

    /**
     * @see ij.plugin.filter.PlugInFilter#run(ij.process.ImageProcessor)
     */
    @Override
    public void run(ImageProcessor ip) {
        // Denoise the image
        ImageStack denoised = this.denoiseImage(m_image.getImageStack(), m_patchSize, m_windowSize, m_blockReductionFactor, m_decayControl, m_lowLight, m_3D);

        // Display output
        ImagePlus output = new ImagePlus("Denoised - " + m_image.getTitle(), denoised);
        output.setCalibration(m_image.getCalibration());
        output.show();
    }

    public ImageStack denoiseImage(ImageStack input, int patchSize, int windowSize, int blockReductionFactor, double decayControl, boolean lowLightConditions, boolean true3D) {
        // Stabilise variance if needed
        ImageStack correctedInput;
        Matrix parameters = new Matrix(2,1);

        if(lowLightConditions) {
            parameters = this.estimatePoissonGaussianParameters(input);

            correctedInput = this.varianceStabilisationPoissonGaussian(input, parameters.get(0,0), parameters.get(1,0));

            // Remove NaNs
            IJ.run(new ImagePlus("Corrected",correctedInput), "Remove NaNs...", "radius=3 stack");
        }
        else { // !lowLightConditions
            correctedInput = input;
        }

//        (new ImagePlus("corr", correctedInput)).show();
//        (new ImagePlus("local var",this.computeLocalGaussianVariance(correctedInput))).show();
//        IJ.showMessage(""+this.estimateGlobalGaussianVariance(correctedInput, this.computeLocalGaussianVariance(correctedInput)));

        // Add margins
        int margin = windowSize/2;
        correctedInput = this.addMargin(correctedInput, margin);

        // Compute denoised image
        ImageStack denoised;

        if(true3D) {
            denoised = this.computeBlockWiseNLMGaussianDenoising(correctedInput,
                    patchSize,
                    windowSize,
                    blockReductionFactor+1,
                    this.estimateGlobalGaussianVariance(correctedInput, this.computeLocalGaussianVariance(correctedInput)),
                    decayControl
            );
        }
        else { // !true3D
            denoised = new ImageStack(correctedInput.getWidth(),correctedInput.getHeight());

            for(int z = 0; z < correctedInput.getSize(); z++) {
                ImageStack currentSlice = (new ImagePlus("",correctedInput.getProcessor(z+1))).getImageStack();

                denoised.addSlice(this.computeBlockWiseNLMGaussianDenoising(currentSlice,
                        patchSize,
                        windowSize,
                        blockReductionFactor,
                        this.estimateGlobalGaussianVariance(currentSlice, this.computeLocalGaussianVariance(currentSlice)),
                        decayControl
                ).getProcessor(1));
            }
        }

        // Remove margins
        denoised = this.removeMargin(denoised, margin);

        // Inverse stabilisation if needed
        if(lowLightConditions) {
            denoised = this.inverseVarianceStabilisationPoissonGaussian(denoised, parameters.get(0,0), parameters.get(1,0));
            denoised = this.convertToBitDepth(denoised, input.getBitDepth());
        }

        return denoised;
    }

    /**
     * Get extent of a patch given a point, size of patch and size of image.
     * The extent is defined by four integers : min and max values of x coordinate
     * and min and max values of y coordinate.
     * @param x Center point of the patch (X coords)
     * @param y Center point of the patch (Y coords)
     * @param patchSize Size of the patch
     * @param width Width of the image
     * @param height Height of the image
     * @return An array containing four integers defining the extent
     */
    private int[] getExtent(int x, int y, int z, int patchSize, int width, int height, int depth) {
        // Initialize extent
        int[] extent = new int[6];

        // Fill it
        extent[0] = x - patchSize/2;
        extent[1] = x + patchSize/2;
        extent[2] = y - patchSize/2;
        extent[3] = y + patchSize/2;
        extent[4] = z - patchSize/2;
        extent[5] = z + patchSize/2;

        // Check boundary and crop if necessary
        if(extent[0] < 0)
            extent[0] = 0;

        if(extent[1] > width)
            extent[1] = width-1;

        if(extent[2] < 0)
            extent[2] = 0;

        if(extent[3] > height)
            extent[3] = height-1;

        if(extent[4] < 0)
            extent[4] = 0;

        if(extent[5] > depth)
            extent[5] = depth-1;

        // Return the extent
        return extent;
    }

    /**
     * Compute the non-local weight of the non-local means algorithm.
     * @param ip Image containing intensity values.
     * @param firstPointExtent Extent of the first patch.
     * @param secondPointExtent Extent of the second patch.
     * @param decayControl Control the decay of the weighting function.
     * @return The weight of the non-local means algorithm.
     */
    private double getNLWeight(ImageStack ip, int[] firstPointExtent, int[] secondPointExtent, double decayControl) {
        return Math.exp(- this.getEuclidianSquaredDistance(ip, firstPointExtent, secondPointExtent) / decayControl);
    }

    /**
     * Compute the euclidian squared distance between two vectors.
     * @param ip Image containing intensity values.
     * @param firstPointExtent Extent of the first patch.
     * @param secondPointExtent Extent of the second patch.
     * @return The squared euclidian distance between the two vectors of intensities.
     */
    private double getEuclidianSquaredDistance(ImageStack ip, int[] firstPointExtent, int[] secondPointExtent) {
        double distance = 0.0;

        for (int firstZ = firstPointExtent[4], secondZ = secondPointExtent[4]; firstZ <= firstPointExtent[5] && secondZ <= secondPointExtent[5]; firstZ++, secondZ++) {
            for (int firstY = firstPointExtent[2], secondY = secondPointExtent[2]; firstY <= firstPointExtent[3] && secondY <= secondPointExtent[3]; firstY++, secondY++) {
                for (int firstX = firstPointExtent[0], secondX = secondPointExtent[0]; firstX <= firstPointExtent[1] && secondX <= secondPointExtent[1]; firstX++, secondX++) {
                    double difference = ip.getVoxel(firstX, firstY, firstZ) - ip.getVoxel(secondX, secondY, secondZ);
                    distance += difference * difference;
                }
            }
        }

        return distance;
    }

    /**
     * Compute the blockwise NLM denoising image with white additive Gaussian assumption of the noise.
     * Buades et al. "A Review of Image Denoising Algorithms, with a New One", Multiscale Modeling Simulation, 2005, vol. 4, pp. 490-530
     * @param input Input image.
     * @param blockSize Size of the block.
     * @param windowSize Size of the search window.
     * @param globalVariance Variance of the white additive Gaussian noise.
     * @param blockStep Step of blockwise algorithm.
     * @param decayControl Control the decay of the weighting function.
     * @return The denoised image.
     */
    private ImageStack computeBlockWiseNLMGaussianDenoising(ImageStack input, int blockSize, int windowSize, int blockStep, double globalVariance, double decayControl) {
        // Get image constants
        int  width = input.getWidth();
        int height = input.getHeight();
        int  depth = input.getSize();

        // Pre-compute parameters
        int numberOfPixelsPerBlock;

        if(depth > 1) {
            numberOfPixelsPerBlock = blockSize * blockSize * blockSize;
            m_maxProgress = (width * height * depth) / (blockStep * blockStep * blockStep);
        }
        else { // depth == 1
            numberOfPixelsPerBlock = blockSize * blockSize;
            m_maxProgress = (width * height) / (blockStep * blockStep);
        }

        double decay = 2.0 * numberOfPixelsPerBlock * globalVariance * decayControl;

        // Initialise output
        ImageStack    output = ImageStack.create(width, height, depth, input.getBitDepth());
        ImageStack tmpOutput = ImageStack.create(width, height, depth, 32);
        ImageStack  tmpCount = ImageStack.create(width, height, depth, 8);

        // Initialise progress bar
        m_currentProgress = 0;
        IJ.showProgress(m_currentProgress, m_maxProgress);

        // For each pixel of image, compute the NLM corrected intensity
        ArrayList< BlockwiseNLMGaussianParallelInput > pInputs = new ArrayList< BlockwiseNLMGaussianParallelInput >();

        for (int z = 0; z < depth; z+=blockStep) {
            for (int y = 0; y < height; y+=blockStep) {
                for (int x = 0; x < width; x+=blockStep) {
                    BlockwiseNLMGaussianParallelInput pInput = new BlockwiseNLMGaussianParallelInput();

                    pInput.parent         = this;
                    pInput.x              = x;
                    pInput.y              = y;
                    pInput.z              = z;
                    pInput.blockSize      = blockSize;
                    pInput.windowSize     = windowSize;
                    pInput.width          = width;
                    pInput.height         = height;
                    pInput.depth          = depth;
                    pInput.decayControl   = decay;
                    pInput.input          = input;
                    pInput.globalVariance = globalVariance;
                    pInput.output         = tmpOutput;
                    pInput.count          = tmpCount;
                    pInput.numberOfPixelsPerBlock = numberOfPixelsPerBlock;
                    pInput.blockStep      = blockStep;

                    pInputs.add(pInput);
                }
            }
        }

        // Run the threads
        try {
            this.parallelProcessBlockwiseNLMGaussianDenoising(pInputs);
        }
        catch (Exception e){
            IJ.showMessage(e.getMessage());
        }

        // Put the final intensities
        float[]  tmpCountIntensities = tmpCount.getVoxels(0, 0, 0, width, height, depth, null);
        float[] tmpOutputIntensities = tmpOutput.getVoxels(0, 0, 0, width, height, depth, null);
        float[]    outputIntensities = new float[tmpOutputIntensities.length];

        for(int i = 0; i < tmpCountIntensities.length; i++) {
            outputIntensities[i] = tmpOutputIntensities[i] / tmpCountIntensities[i];
        }

        output.setVoxels(0, 0, 0, width, height, depth, outputIntensities);

        return output;
    }

    /**
     * Parameters input for parallel processing (blockwise).
     */
    private class BlockwiseNLMGaussianParallelInput {
        NLM_Denoising parent;
        int           x,y,z;
        int           blockSize;
        int           windowSize;
        int           width,height,depth;
        double        decayControl;
        ImageStack    input;
        double        globalVariance;
        ImageStack    output;
        ImageStack    count;
        int           numberOfPixelsPerBlock;
        int           blockStep;
    }

    /**
     * Parallel processing of blockwise NLM algorithm.
     * @see this.computeBlockWiseNLMGaussianDenoising
     * @param inputs Input parameters.
     * @throws Exception
     */
    private void parallelProcessBlockwiseNLMGaussianDenoising(ArrayList< BlockwiseNLMGaussianParallelInput > inputs) throws Exception {
        // Create a service for threads (same number as available processors)
        int threads = Runtime.getRuntime().availableProcessors() + 1;
        ExecutorService service = Executors.newFixedThreadPool(threads);

        // For each input, do a parallel process using service
        for(final BlockwiseNLMGaussianParallelInput input : inputs) {
            final Runnable runnable = new Runnable() {
                @Override
                public void run() {
                    // Initialise weighted sum and normalisation constant
                    float[] weightedSum = new float[input.numberOfPixelsPerBlock];
                    float[] weightedSum2 = new float[input.numberOfPixelsPerBlock];
                    float normConstant = 0.0f;

                    // Get patch extent of point under denoising estimate
                    int[] estimatePointExtent = input.parent.getExtent(input.x, input.y, input.z, input.blockSize, input.width, input.height, input.depth);

                    // Get search window extent of point under denoising estimate
                    int[] searchExtent = input.parent.getExtent(input.x, input.y, input.z, input.windowSize, input.width, input.height, input.depth);

                    // For each point in search window
                    float maxWeight = 0.0f;

                    for (int searchZ = searchExtent[4]; searchZ <= searchExtent[5]; searchZ+=input.blockStep) {
                        for (int searchY = searchExtent[2]; searchY <= searchExtent[3]; searchY+=input.blockStep) {
                            for (int searchX = searchExtent[0]; searchX <= searchExtent[1]; searchX+=input.blockStep) {
                                if (searchX != input.x || searchY != input.y) {
                                    // Get patch extent of second point
                                    int[] lookPointExtent = input.parent.getExtent(searchX, searchY, searchZ, input.blockSize, input.width, input.height, input.depth);

                                    // Compute the NLM weight with this particular point
                                    float weight = (float) input.parent.getNLWeight(input.input, estimatePointExtent, lookPointExtent, input.decayControl);

                                    // Compute the estimate
                                    float[] currentIntensity = input.input.getVoxels(
                                            lookPointExtent[0], lookPointExtent[2], lookPointExtent[4],
                                            lookPointExtent[1]-lookPointExtent[0]+1, lookPointExtent[3]-lookPointExtent[2]+1, lookPointExtent[5]-lookPointExtent[4]+1,
                                            null
                                    );

                                    for(int i = 0; i < currentIntensity.length; i++) {
                                        weightedSum[i]  += weight * currentIntensity[i];
                                        weightedSum2[i] += weight * currentIntensity[i] * currentIntensity[i];
                                    }

                                    normConstant += weight;

                                    // Look for the maximal weight
                                    if (weight > maxWeight) {
                                        maxWeight = weight;
                                    }
                                }
                            }
                        }
                    }

                    // Add the contribution of the noisy current pixel
                    int x = estimatePointExtent[0], y = estimatePointExtent[2], z = estimatePointExtent[4];
                    int w = estimatePointExtent[1]-estimatePointExtent[0]+1, h = estimatePointExtent[3]-estimatePointExtent[2]+1, d = estimatePointExtent[5]-estimatePointExtent[4]+1;
                    float[] inputIntensity = input.input.getVoxels(x, y, z, w, h, d, null);

                    for(int i = 0; i < inputIntensity.length; i++) {
                        weightedSum[i]  += maxWeight * inputIntensity[i];
                        weightedSum2[i] += maxWeight * inputIntensity[i] * inputIntensity[i];
                    }

                    normConstant += maxWeight;

                    synchronized (input.parent) {
                        // Correct the intensity estimate when high local variance
                        float[] outputIntensities = input.output.getVoxels(x, y, z, w, h, d, null);
                        float[] counts = input.count.getVoxels(x, y, z, w, h, d, null);

                        for(int i = 0; i < outputIntensities.length; i++) {
                            float      intensityEstimate = weightedSum[i] / normConstant;
                            float     intensity2Estimate = weightedSum2[i] / normConstant;

                            if(m_keepDetails) {
                                float localVarianceIntensity = intensity2Estimate - intensityEstimate * intensityEstimate;

                                outputIntensities[i] += intensityEstimate + Math.max(0.0, (localVarianceIntensity - input.globalVariance) / localVarianceIntensity) * (inputIntensity[i] - intensityEstimate);
                            }
                            else { // !m_keepDetails
                                outputIntensities[i] += intensityEstimate;
                            }

                            counts[i]++;
                        }

                        input.output.setVoxels(x, y, z, w, h, d, outputIntensities);
                        input.count.setVoxels(x, y, z, w, h, d, counts);

                        // Update progress
                        IJ.showProgress(++m_currentProgress, m_maxProgress);
                    }
                }
            };

            // Add to service
            service.submit(runnable);
        }

        // Stop the threads service
        service.shutdown();
        if(!service.awaitTermination(30, TimeUnit.MINUTES)){
            throw new Exception("Not waiting anymore the threads to finish !");
        }
    }

    /**
     * Compute the local variance using pseudo residuals of image.
     * @param input Input image.
     * @return Image of variance.
     */
    public ImageStack computeLocalGaussianVariance(ImageStack input) {
        // Get image constants
        int  width = input.getWidth();
        int height = input.getHeight();
        int  depth = input.getSize();

        // Pre-compute constants
        double neighborhoodSumNormalisationConstant = 1.0 / 6.0;
        double residualNormalisationConstant = Math.sqrt( 6.0 / 7.0 );

        // Initialise output
        ImageStack output = ImageStack.create(width, height, depth, 32);

        // For each pixel, compute the pseudo residuals
        for (int z = 0; z < depth; z++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    // Get extent of neighborhood
                    int[] extent = this.getExtent(x, y, z, 3, width, height, depth);

                    // Compute neighborhood intensities sum
                    double neighborhoodSum = 0.0;

                    for (int extentZ = extent[4]; extentZ <= extent[5]; extentZ++) {
                        for (int extentY = extent[2]; extentY <= extent[3]; extentY++) {
                            for (int extentX = extent[0]; extentX <= extent[1]; extentX++) {
                                neighborhoodSum += input.getVoxel(extentX, extentY, extentZ);
                            }
                        }
                    }

                    // Compute current residuals
                    double residual = residualNormalisationConstant * (input.getVoxel(x,y,z) - neighborhoodSumNormalisationConstant * neighborhoodSum);
                    output.setVoxel(x, y, z, (float) (residual * residual));
                }
            }
        }

        return output;
    }

    /**
     * Estimate the gaussian variance of the input image using the pseudo residuals.
     * Gasser et al. "Residuals variance and residual pattern in nonlinear regression", Biometrika, 1986, vol. 73(3), pp. 625-633
     * @param input Input image.
     * @param localVariances Pseudo residuals image.
     * @return Estimated variance.
     */
    public double estimateGlobalGaussianVariance(ImageStack input, ImageStack localVariances) {
        // Get image constants
        int  width = input.getWidth();
        int height = input.getHeight();
        int  depth = input.getSize();

        // Estimate variance
        double variance = 0.0;
        int numberOfNaNs = 0;

        for (int z = 0; z < depth; z++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double localVariance = localVariances.getVoxel(x, y, z);

                    if (Double.isNaN(localVariance)) {
                        numberOfNaNs++;
                    }
                    else {
                        variance += localVariance;
                    }
                }
            }
        }

        variance /= width * height * depth - numberOfNaNs;

        return variance;
    }

    /**
     * Compute the general Anscombe transform for variance stabilisation (from Poisson/Gaussian to std Gaussian statistics).
     * Murtagh et al. "Image restoration with noise suppression using a multiresolution support", Astron. Astrophys., 1995, vol. 112, pp. 197–189
     * @param input Input image (with Poisson/Gaussian statistics).
     * @param gain Image gain.
     * @param gaussianParameter From the dark current of CCD.
     * @return Transformed image (with std Gaussian statistics).
     */
    public ImageStack varianceStabilisationPoissonGaussian(ImageStack input, double gain, double gaussianParameter) {
        // Get image constants
        int  width = input.getWidth();
        int height = input.getHeight();
        int  depth = input.getSize();

        // Pre-compute constants
        double offset = (3.0/8.0) * gain * gain + gaussianParameter;
        double  coeff = 2.0/gain;

        // Initialise output
        ImageStack output = ImageStack.create(width, height, depth, 32);

        // Compute the transform
        for(int z = 0; z < depth; z++) {
            for(int y = 0; y < height; y++) {
                for(int x = 0; x < width; x++) {
                    output.setVoxel(x,y,z, coeff * Math.sqrt(gain*input.getVoxel(x,y,z) + offset));
                }
            }
        }

        return output;
    }

    /**
     * Compute the general Anscombe inverse transform for variance stabilisation (from Poisson/Gaussian to Gaussian statistics).
     * Murtagh et al. "Image restoration with noise suppression using a multiresolution support", Astron. Astrophys., 1995, vol. 112, pp. 197–189
     * Boulanger et al. "Patch-Based Nonlocal Functional for Denoising Fluorescence Microscopy Image Sequences", 2010, Medical Imaging, IEEE Transactions on, vol. 29, pp. 442-454
     * @param input Input image (with std Gaussian statistics).
     * @param gain Image gain.
     * @param gaussianParameter From the dark current of CCD.
     * @return Transformed image (with Gaussian/Poisson statistics).
     */
    public ImageStack inverseVarianceStabilisationPoissonGaussian(ImageStack input, double gain, double gaussianParameter) {
        // Get image constants
        int  width = input.getWidth();
        int height = input.getHeight();
        int  depth = input.getSize();

        // Pre-compute constants
        double  coeff = gain * gain / 4.0;
        double offset = - 3.0 * gain * gain / 8.0 - gaussianParameter;
        double normco = 1.0 / gain;

        // Initialise output
        ImageStack output = ImageStack.create(width, height, depth, 32);

        // Compute the inverse transform
        for(int z = 0; z < depth; z++) {
            for(int y = 0; y < height; y++) {
                for(int x = 0; x < width; x++) {
                    double           value = input.getVoxel(x, y, z);
                    double   biasedInverse = normco * (coeff*value*value + offset);
                    double unbiasedInverse = biasedInverse + (1.0/4.0) * (1.0 - Math.exp(-1.3*biasedInverse));

//                    output.setVoxel(x,y,z, (unbiasedInverse >= 0.0) ? unbiasedInverse : 0.0);
                    output.setVoxel(x,y,z, unbiasedInverse);
                }
            }
        }

        return output;
    }

    /**
     * Compute the Laplacian residuals for input image (more robust for non-Gaussian than pseudo-residuals).
     * @param input Input image.
     * @return Laplacian residuals image.
     */
    public ImageStack computeLaplacianResiduals(ImageStack input) {
        // Compute image constants
        int  width = input.getWidth();
        int height = input.getHeight();
        int  depth = input.getSize();

        int d = 2;
        int l = 2*d + 1;

        double coeff = 1.0 / Math.sqrt(l*l + l);

        // Initialise residuals image
        ImageStack residuals = ImageStack.create(width, height, depth, 32);

        // Compute residuals
        for (int z = 0; z < depth; z++) {
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    double laplacian = l * input.getVoxel(x,y,z) - (input.getVoxel(x+1,y,z) + input.getVoxel(x-1,y,z) + input.getVoxel(x,y+1,z) + input.getVoxel(x,y-1,z));

                    residuals.setVoxel(x, y, z, coeff * laplacian);
                }
            }
        }

        return residuals;
    }

    /**
     * Compute means and variance of Poisson/Gaussian statistics image using a uniform variance blocks method (recursive method).
     * Boulanger et al. "Patch-Based Nonlocal Functional for Denoising Fluorescence Microscopy Image Sequences", 2010, Medical Imaging, IEEE Transactions on, vol. 29, pp. 442-454
     * @param input Input image.
     * @param residuals Laplacian residuals image.
     * @param extent Initial extent (or block).
     * @return An array of means and variances of blocks.
     */
    public ArrayList< double[] > getMeanAndVarianceFromUniformBlocksOfVariance(ImageProcessor input, ImageProcessor residuals, int[] extent) {
        // Compute mean
        double intensities_mean = 0.0;
        double   residuals_mean = 0.0;
        int      numberOfPixels = 0;

        for (int y = extent[2]; y < extent[3]; y++) {
            for (int x = extent[0]; x < extent[1]; x++) {
                intensities_mean += input.getf(x,y);
                residuals_mean   += residuals.getf(x,y);
                numberOfPixels++;
            }
        }

        intensities_mean /= numberOfPixels;
        residuals_mean   /= numberOfPixels;

        // Compute variance
        double intensities_variance = 0.0;
        double   residuals_variance = 0.0;

        for (int y = extent[2]; y < extent[3]; y++) {
            for (int x = extent[0]; x < extent[1]; x++) {
                double diff = input.getf(x,y) - intensities_mean;
                intensities_variance += diff * diff;

                diff = residuals.getf(x,y) - residuals_mean;
                residuals_variance += diff * diff;
            }
        }

        intensities_variance /= numberOfPixels - 1;
        residuals_variance   /= numberOfPixels - 1;

        double fisherTest = Math.min(intensities_variance,residuals_variance) / Math.max(intensities_variance,residuals_variance);

        ArrayList< double[] > list = new ArrayList< double[] >();

        int size_half_x = (extent[1] - extent[0]) / 2;
        int size_half_y = (extent[3] - extent[2]) / 2;

        if (fisherTest >= 0.8 || size_half_x < 2 || size_half_y < 2) {
            if (4*size_half_x*size_half_y >= 20) {
                double[] couple = {intensities_mean, intensities_variance};
                list.add(couple);
            }
        }
        else {
            int middle_point_x = extent[0] + size_half_x;
            int middle_point_y = extent[2] + size_half_y;

            int[] new_extent_1 = new int[4];
            new_extent_1[0] = extent[0]; new_extent_1[1] = middle_point_x;
            new_extent_1[2] = extent[2]; new_extent_1[3] = middle_point_y;

            int[] new_extent_2 = new int[4];
            new_extent_2[0] = middle_point_x; new_extent_2[1] = extent[1];
            new_extent_2[2] = extent[2];      new_extent_2[3] = middle_point_y;

            int[] new_extent_3 = new int[4];
            new_extent_3[0] = extent[0];      new_extent_3[1] = middle_point_x;
            new_extent_3[2] = middle_point_y; new_extent_3[3] = extent[3];

            int[] new_extent_4 = new int[4];
            new_extent_4[0] = middle_point_x; new_extent_4[1] = extent[1];
            new_extent_4[2] = middle_point_y; new_extent_4[3] = extent[3];

            list.addAll(this.getMeanAndVarianceFromUniformBlocksOfVariance(input, residuals, new_extent_1));
            list.addAll(this.getMeanAndVarianceFromUniformBlocksOfVariance(input, residuals, new_extent_2));
            list.addAll(this.getMeanAndVarianceFromUniformBlocksOfVariance(input, residuals, new_extent_3));
            list.addAll(this.getMeanAndVarianceFromUniformBlocksOfVariance(input, residuals, new_extent_4));
        }

        return list;
    }

    /**
     * Estimate the Poisson/Gaussian parameters of input noise.
     * Boulanger et al. "Patch-Based Nonlocal Functional for Denoising Fluorescence Microscopy Image Sequences", 2010, Medical Imaging, IEEE Transactions on, vol. 29, pp. 442-454
     * @param input Input image.
     * @return Parameters of Poisson/Gaussian noise.
     */
    private Matrix estimatePoissonGaussianParameters(ImageStack input) {
        // Compute the laplacian residuals
        ImageStack residuals = this.computeLaplacianResiduals(input);

        // For each slice, get mean and variance of uniform regions
        int[] extent = new int[4];
        extent[0] = 0; extent[1] = input.getWidth();
        extent[2] = 0; extent[3] = input.getHeight();

        ArrayList< double[] > list = this.getMeanAndVarianceFromUniformBlocksOfVariance(input.getProcessor(1), residuals.getProcessor(1), extent);

        for(int z = 1; z < input.getSize(); z++) {
            list.addAll(this.getMeanAndVarianceFromUniformBlocksOfVariance(input.getProcessor(z + 1), residuals.getProcessor(z + 1), extent));
        }

        // Estimate the parameters (least squares estimation)
        Matrix X = new Matrix(list.size(),2,1);
        for(int i = 0; i < list.size(); i++) {
            X.set(i,0, list.get(i)[0]);
        }

        Matrix Y = new Matrix(list.size(),1);
        for(int i = 0; i < list.size(); i++) {
            Y.set(i,0, list.get(i)[1]);
        }

        Matrix beta;

        try {
            beta = (X.transpose().times(X)).inverse().times(X.transpose()).times(Y);
        }
        catch (Exception e) {
            beta = new Matrix(2,1);
            beta.set(0,0, 1);
            beta.set(1,0, 0);
        }

        return beta;
    }

    /**
     * Add a margin to the image as a mirror (management of the image borders).
     * @param input Input image.
     * @param margin Margin size.
     * @return Input image with a mirror margin added.
     */
    private ImageStack addMargin(ImageStack input, int margin) {
        // Compute image constants
        int  width = input.getWidth();
        int height = input.getHeight();
        int  depth = input.getSize();

        int  oWidth = width + 2*margin;
        int oHeight = height + 2*margin;
        int  oDepth = (depth > 1) ? depth + 2*margin : 1;

        // Initialise new image with margin (mirror)
        ImageStack output = ImageStack.create(oWidth, oHeight, oDepth, input.getBitDepth());

        if (depth > 1) {
            // Fill the image
            for (int zi = 0, zo = margin; zi < depth; zi++, zo++) {
                for (int yi = 0, yo = margin; yi < height; yi++, yo++) {
                    for (int xi = 0, xo = margin; xi < width; xi++, xo++) {
                        output.setVoxel(xo, yo, zo, input.getVoxel(xi, yi, zi));
                    }
                }
            }

            // Fill the margin
            // z min
            for (int zo = 0; zo < margin; zo++) {
                for (int yo = margin; yo < height + margin; yo++) {
                    for (int xo = margin; xo < width + margin; xo++) {
                        output.setVoxel(xo, yo, zo, output.getVoxel(xo, yo, 2 * margin - 1 - zo));
                    }
                }
            }

            // z max
            for (int zo = depth + margin; zo < oDepth; zo++) {
                for (int yo = margin; yo < height + margin; yo++) {
                    for (int xo = margin; xo < width + margin; xo++) {
                        output.setVoxel(xo, yo, zo, output.getVoxel(xo, yo, depth + oDepth - zo - 1));
                    }
                }
            }

            // y min
            for (int yo = 0; yo < margin; yo++) {
                for (int zo = 0; zo < oDepth; zo++) {
                    for (int xo = margin; xo < width + margin; xo++) {
                        output.setVoxel(xo, yo, zo, output.getVoxel(xo, 2 * margin - 1 - yo, zo));
                    }
                }
            }

            // y max
            for (int yo = height + margin; yo < oHeight; yo++) {
                for (int zo = 0; zo < oDepth; zo++) {
                    for (int xo = margin; xo < width + margin; xo++) {
                        output.setVoxel(xo, yo, zo, output.getVoxel(xo, height + oHeight - yo - 1, zo));
                    }
                }
            }

            // x min
            for (int xo = 0; xo < margin; xo++) {
                for (int zo = 0; zo < oDepth; zo++) {
                    for (int yo = 0; yo < oHeight; yo++) {
                        output.setVoxel(xo, yo, zo, output.getVoxel(2 * margin - 1 - xo, yo, zo));
                    }
                }
            }

            // x max
            for (int xo = width + margin; xo < oWidth; xo++) {
                for (int zo = 0; zo < oDepth; zo++) {
                    for (int yo = 0; yo < oHeight; yo++) {
                        output.setVoxel(xo, yo, zo, output.getVoxel(width + oWidth - xo - 1, yo, zo));
                    }
                }
            }
        }
        else { // depth == 1
            // Fill the image
            for (int yi = 0, yo = margin; yi < height; yi++, yo++) {
                for (int xi = 0, xo = margin; xi < width; xi++, xo++) {
                    output.setVoxel(xo, yo, 0, input.getVoxel(xi, yi, 0));
                }
            }

            // Fill the margin
            // y min
            for (int yo = 0; yo < margin; yo++) {
                for (int xo = margin; xo < width + margin; xo++) {
                    output.setVoxel(xo, yo, 0, output.getVoxel(xo, 2 * margin - 1 - yo, 0));
                }
            }

            // y max
            for (int yo = height + margin; yo < oHeight; yo++) {
                for (int xo = margin; xo < width + margin; xo++) {
                    output.setVoxel(xo, yo, 0, output.getVoxel(xo, height + oHeight - yo - 1, 0));
                }
            }

            // x min
            for (int xo = 0; xo < margin; xo++) {
                for (int yo = 0; yo < oHeight; yo++) {
                    output.setVoxel(xo, yo, 0, output.getVoxel(2 * margin - 1 - xo, yo, 0));
                }
            }

            // x max
            for (int xo = width + margin; xo < oWidth; xo++) {
                for (int yo = 0; yo < oHeight; yo++) {
                    output.setVoxel(xo, yo, 0, output.getVoxel(width + oWidth - xo - 1, yo, 0));
                }
            }
        }

        return output;
    }

    /**
     * Remove the margin added on image.
     * @see this.addMargin
     * @param input Input image with margin.
     * @param margin Margin size.
     * @return Image without margin.
     */
    private ImageStack removeMargin(ImageStack input, int margin) {
        // Compute image constants
        int  width = input.getWidth();
        int height = input.getHeight();
        int  depth = input.getSize();

        int  oWidth = width - 2*margin;
        int oHeight = height - 2*margin;
        int  oDepth = (depth > 1) ? depth - 2*margin : 1;

        // Initialise new image with margin (mirror)
        ImageStack output = ImageStack.create(oWidth, oHeight, oDepth, input.getBitDepth());

        if (depth > 1) {
            // Fill the image
            for (int zi = margin, zo = 0; zi < depth - margin; zi++, zo++) {
                for (int yi = margin, yo = 0; yi < height - margin; yi++, yo++) {
                    for (int xi = margin, xo = 0; xi < width - margin; xi++, xo++) {
                        output.setVoxel(xo, yo, zo, input.getVoxel(xi, yi, zi));
                    }
                }
            }
        }
        else { // depth == 1
            // Fill the image
            for (int yi = margin, yo = 0; yi < height - margin; yi++, yo++) {
                for (int xi = margin, xo = 0; xi < width - margin; xi++, xo++) {
                    output.setVoxel(xo, yo, 0, input.getVoxel(xi, yi, 0));
                }
            }
        }

        return output;
    }

    /**
     * Convert input to specified bit depth.
     * @param input Input image.
     * @param bitDepth Bit depth to convert to.
     * @return Converted image.
     */
    private ImageStack convertToBitDepth(ImageStack input, int bitDepth) {
        int  width = input.getWidth();
        int height = input.getHeight();
        int  depth = input.getSize();

        ImageStack output = ImageStack.create(width, height, depth, bitDepth);

        for(int z = 0; z < depth; z++) {
            for(int y = 0; y < height; y++) {
                for(int x = 0; x < width; x++) {
                    output.setVoxel(x,y,z, input.getVoxel(x,y,z));
                }
            }
        }

        return output;
    }

//    /**
//     * Event listener of the dialog gui.
//     * @see ij.gui.DialogListener#dialogItemChanged(ij.gui.GenericDialog, java.awt.AWTEvent)
//     * @param gui Graphical user interface object who raise an event
//     * @param e Event raised
//     * @return True if the parameters are allowed, false otherwise
//     */
//    @Override
//    public boolean dialogItemChanged(GenericDialog gui, AWTEvent e)
//    {
//        boolean dialogOk = true;
//
//        if(e != null && e.paramString().equals("TEXT_VALUE_CHANGED"))
//        {
//            // Get fields
//            int        patchSize = (int)gui.getNextNumber();
//            int       windowSize = (int)gui.getNextNumber();
//            double noiseVariance = gui.getNextNumber();
//            double         decay = gui.getNextNumber();
//            int        blockStep = (int)gui.getNextNumber();
//
//            // Activate or deactivate Ok button
//            if(patchSize < 3 ||
//                    windowSize <= patchSize ||
//                    Double.compare(noiseVariance,0.0) < 0 ||
//                    (patchSize-1 <= blockStep)
//                    ) {
//                dialogOk = false;
//            }
//        }
//        // else any other event
//
//        return dialogOk;
//    }

    /**
     * Main method for debugging.
     *
     * For debugging, it is convenient to have a method that starts ImageJ, loads an
     * image and calls the plugin, e.g. after setting breakpoints.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        // set the plugins.dir property to make the plugin appear in the Plugins menu
        Class<?> clazz = NLM_Denoising.class;
        String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
        String pluginsDir = url.substring(5, url.length() - clazz.getName().length() - 6);
        System.setProperty("plugins.dir", pluginsDir);

        // start ImageJ
        new ImageJ();

        // Testing

    }
}
