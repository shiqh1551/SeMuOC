class Config(object):
    def __init__(self):
        
        self.dataset = 'WADI'

        self.input_channels = 127
        self.window_size = 128
        self.time_step = 64

        
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
            "WADI is a water distribution testbed with sensors and actuators "
            "monitoring tanks, pumps, valves, flow, pressure, and water movement. "
            "Normal behavior should preserve hydraulic consistency and stable "
            "relations between control actions and sensor responses."
        )
        self.variable_desc = {
            f"Channel-{idx}": f"WADI sensor or actuator variable {idx}"
            for idx in range(self.input_channels)
        }
        self.anomaly_texts = [
            "Anomaly pattern: valve or pump state changes but flow or pressure does not respond.",
            "Anomaly pattern: tank level rises or falls inconsistently with inlet and outlet flow.",
            "Anomaly pattern: pressure or flow drops abruptly while upstream controls remain active.",
            "Anomaly pattern: a sensor drifts slowly without corresponding actuator changes.",
            "Anomaly pattern: related hydraulic variables become inconsistent.",
            "Anomaly pattern: a stable water-distribution signal becomes volatile with local bursts.",
        ]

        
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

        
        self.num_epoch = 5
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
