class Config(object):
    def __init__(self):
        
        self.dataset = 'ESA'

        self.input_channels = 6
        self.window_size = 128
        self.time_step = 128

        
        self.d_model = 64
        self.tcn_layers = 3
        self.attention_heads = 4
        self.dropout = 0.1
        self.score_alpha = 1.0
        self.score_beta = 1.0
        self.text_model = "intfloat/e5-base-v2"
        self.allow_hash_text_encoder = False
        self.text_embedding_cache = True
        self.text_cache_on_device = True
        self.domain_text = (
            "ESA contains multivariate spacecraft or engineering telemetry. "
            "Variables describe sensor, subsystem, and operational measurements. "
            "Normal behavior should respect temporal continuity, subsystem coupling, "
            "and stable operating regimes."
        )
        self.variable_desc = {
            f"Channel-{idx}": f"ESA telemetry variable {idx}"
            for idx in range(self.input_channels)
        }
        self.anomaly_texts = [
            "Anomaly pattern: telemetry suddenly spikes or drops without matching subsystem behavior.",
            "Anomaly pattern: a sensor drifts gradually away from its normal operating regime.",
            "Anomaly pattern: coupled telemetry variables become inconsistent with each other.",
            "Anomaly pattern: expected spacecraft or subsystem periodic behavior changes unexpectedly.",
            "Anomaly pattern: a stable telemetry channel becomes volatile with local bursts.",
            "Anomaly pattern: multiple subsystem indicators show conflicting operational states.",
        ]

        # SemSphere one-class branch (AOC-style projection and fixed center)
        self.project_dim = 32
        self.oc_hidden_dim = None
        self.oc_pooling = "flatten"
        self.center_eps = 0.1
        self.center_normal_only = True

        
        self.use_infonce = True
        self.lambda_oc = 0.1
        self.lambda_align = 0.1
        self.lambda_anom = 0.1
        self.lambda_recon = 1.0
        self.anomaly_similarity_margin = 0.2

        
        self.num_epoch = 50
        self.early_stop = 20
        self.save_diagnostics = True
        self.keep_checkpoints = True

        
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.weight = 5e-4

        
        self.batch_size = 128
        self.drop_last = False

        
        self.detect_nu = 0.0005
        self.threshold_determine = 'floating'
