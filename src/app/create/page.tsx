'use client';

import React, { useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { ChevronDown, ChevronUp, Copy, FileJson2, Plus, Trash2 } from "lucide-react";

type AlgorithmSource = "builtin" | "container";

type StorageProvider = "aws" | "minio" | "gcs" | "azure" | "custom";

type Channel = {
  id: string;
  channelName: string;
  sourceType: "object-storage" | "upload";
  storageProvider?: StorageProvider;
  endpoint?: string;
  bucket?: string;
  prefix?: string;
  region?: string;
  uploadFileName?: string;
  contentType?: string;
  compressionType?: "None" | "Gzip";
  distribution?: "FullyReplicated" | "ShardedByKey";
  recordWrapperType?: "None" | "RecordIO";
  inputModeOverride?: "File" | "Pipe";
};

type TrainingJobForm = {
  jobName: string;
  priority: number;
  region: string;
  algorithm: {
    source: AlgorithmSource;
    algorithmName?: string;
    imageUri?: string;
    enableMetrics: boolean;
  };
  resources: {
    instanceResources: {
      cpuCores: number;
      memoryGiB: number;
      gpuCount: number;
    };
    instanceCount: number;
    volumeSizeGB: number;
    distributed: {
      enabled: boolean;
      strategy: "mpi" | "ps";
      worldSize?: number;
      processesPerHost?: number;
    };
  };
  stoppingCondition: { maxRuntimeSeconds: number };
  inputDataConfig: Channel[];
  outputDataConfig: { artifactUri: string };
};

type JobStatus = "Pending" | "Running" | "Succeeded" | "Failed";

type StoredJob = {
  id: string;
  algorithm: string;
  createdAt: number;
  priority: number;
  status: JobStatus;
  pendingUntil?: number;
};

const builtinAlgorithms = [
  { id: "xgboost", name: "XGBoost" },
  { id: "lightgbm", name: "LightGBM" },
  { id: "tensorflow-cnn", name: "TensorFlow CNN" },
  { id: "tensorflow-transformer", name: "TensorFlow Transformer" },
  { id: "tf-distributed", name: "TensorFlow Distributed" },
  { id: "horovod-mpi", name: "Horovod (MPI)" },
  // { id: "deepspeed-mpi", name: "DeepSpeed (MPI)" },
  { id: "deepspeed-zero3", name: "DeepSpeed" },
  { id: "jax-pjit", name: "JAX PJIT" },
  { id: "kfp-mpijob", name: "Kubeflow MPI" },
  { id: "kfp-jaxjob", name: "Kubeflow JAX" },
  { id: "kfp-tfjob", name: "Kubeflow TF" },
  { id: "kfp-xgboost", name: "Kubeflow XGBoost" },
  { id: "torch-mpi", name: "PyTorch MPI" },
];

const DEFAULT_STORAGE_PROVIDER: StorageProvider = "minio";

const storageProviders = [
  { id: "aws", label: "AWS Object Store" },
  { id: "minio", label: "MinIO" },
  { id: "gcs", label: "Google Cloud Storage" },
  { id: "azure", label: "Azure Blob Storage" },
  { id: "custom", label: "Custom Object Store" },
] as const;

const storageProviderLabels: Record<StorageProvider, string> = {
  aws: "AWS Object Store",
  minio: "MinIO",
  gcs: "Google Cloud Storage",
  azure: "Azure Blob Storage",
  custom: "Custom Object Store",
};

const INPUT_MODE_NONE = "none" as const;
const REGION = "us-east-1";
const RAW_BASE_PATH = (process.env.NEXT_PUBLIC_BASE_PATH ?? process.env.NEXT_BASE_PATH ?? "").trim();
const BASE_PATH = RAW_BASE_PATH ? RAW_BASE_PATH.replace(/\/+$/, "") : "";
const LIST_ROUTE = "/training-job";
const API_JOBS_ENDPOINT = BASE_PATH ? `${BASE_PATH}/api/jobs` : "/api/jobs";

function generateJobName(prefix = "train") {
  const dt = new Date();
  const stamp = dt.toISOString().replace(/[-:T.Z]/g, "").slice(0, 14);
  const random = Math.random().toString(36).slice(2, 6);
  return `${prefix}-${stamp.toLowerCase()}-${random}`;
}

const JOB_NAME_REGEX = /^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$/;

function secondsFromHM(h: number, m: number) {
  const hh = Number.isFinite(h) ? h : 0;
  const mm = Number.isFinite(m) ? m : 0;
  return Math.max(0, Math.floor(hh) * 3600 + Math.floor(mm) * 60);
}

function randomId(prefix = "id") {
  const cryptoObj = typeof globalThis !== "undefined" ? (globalThis as { crypto?: Crypto }).crypto : undefined;
  if (cryptoObj && typeof cryptoObj.randomUUID === "function") {
    return cryptoObj.randomUUID();
  }
  const rand = Math.random().toString(36).slice(2, 10);
  return `${prefix}-${rand}`;
}

function deepClone<T>(x: T): T {
  return JSON.parse(JSON.stringify(x));
}

function formToPayload(form: TrainingJobForm) {
  return {
    jobName: form.jobName,
    priority: form.priority,
    algorithm: {
      source: form.algorithm.source,
      algorithmName: form.algorithm.algorithmName,
      imageUri: form.algorithm.imageUri,
      enableMetrics: form.algorithm.enableMetrics,
    },
    resources: deepClone(form.resources),
    stoppingCondition: deepClone(form.stoppingCondition),
    inputDataConfig: deepClone(form.inputDataConfig),
    outputDataConfig: deepClone(form.outputDataConfig),
  };
}

type JobPayload = ReturnType<typeof formToPayload>;

function validateForm(form: TrainingJobForm) {
  const errors: string[] = [];
  if (!form.jobName || !JOB_NAME_REGEX.test(form.jobName)) {
    errors.push("Job name is required and must match ^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$");
  }
  if (!Number.isFinite(form.priority) || form.priority <= 0 || form.priority > 1000) {
    errors.push("Priority must be greater than 0 and at most 1000.");
  }
  const alg = form.algorithm;
  if (alg.source === "builtin" && !alg.algorithmName) errors.push("Select a built-in algorithm.");
  if (alg.source === "container" && !alg.imageUri) errors.push("Container image URI is required.");
  if (!form.inputDataConfig?.length) errors.push("At least one input channel is required.");
  const hasTrain = form.inputDataConfig.some((c) => c.channelName === "train");
  if (!hasTrain) errors.push("A 'train' channel is required.");
  form.inputDataConfig.forEach((c, idx) => {
    const sourceType = c.sourceType || "object-storage";
    if (!c.channelName) errors.push(`Channel #${idx + 1}: name is required.`);
    if (sourceType === "object-storage") {
      if (!c.storageProvider) errors.push(`Channel '${c.channelName || idx + 1}': provider is required.`);
      if (!c.bucket) errors.push(`Channel '${c.channelName || idx + 1}': bucket/container is required.`);
      if (!c.prefix) errors.push(`Channel '${c.channelName || idx + 1}': prefix/path is required.`);
    } else if (sourceType === "upload") {
      if (!c.uploadFileName) errors.push(`Channel '${c.channelName || idx + 1}': upload file is required.`);
    }
  });
  if (!form.outputDataConfig.artifactUri) errors.push("Output artifact URI is required.");
  return errors;
}

type PersistResponse = { ok: true; filename?: string } | { ok: false };

async function persistPayloadOnServer(payload: JobPayload, job: StoredJob): Promise<PersistResponse> {
  try {
    const response = await fetch(API_JOBS_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ payload, job }),
    });
    if (!response.ok) return { ok: false };
    const parsed = (await response.json()) as { ok?: boolean; filename?: unknown };
    if (parsed?.ok) {
      return {
        ok: true,
        filename: typeof parsed?.filename === "string" ? parsed.filename : undefined,
      };
    }
    return { ok: false };
  } catch (error) {
    console.error("Failed to persist payload", error);
    return { ok: false };
  }
}

function ChannelEditor({ value, onChange }: { value: Channel; onChange: (c: Channel) => void }) {
  const sourceType = value.sourceType || "object-storage";
  const provider = value.storageProvider || DEFAULT_STORAGE_PROVIDER;
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  function updateChannel(partial: Partial<Channel>) {
    onChange({ ...value, ...partial });
  }

  return (
    <div className="grid gap-4 py-2">
      <div className="grid gap-3">
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="grid gap-2">
            <Label>Channel name</Label>
            <Input value={value.channelName} onChange={(e) => updateChannel({ channelName: e.target.value })} placeholder="train" />
          </div>
          <div className="grid gap-2">
            <Label>Content type</Label>
            <Input value={value.contentType || ""} onChange={(e) => updateChannel({ contentType: e.target.value })} placeholder="text/csv" />
          </div>
        </div>
        <div className="grid gap-2">
          <Label>Compression</Label>
          <Select value={value.compressionType || "None"} onValueChange={(v) => updateChannel({ compressionType: v as Channel["compressionType"] })}>
            <SelectTrigger><SelectValue placeholder="Compression" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="None">None</SelectItem>
              <SelectItem value="Gzip">Gzip</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid gap-4">
        <div className="grid gap-2">
          <Label>Source type</Label>
          <RadioGroup
            value={sourceType}
            onValueChange={(v) => {
              const nextSource = v as Channel["sourceType"];
              updateChannel({
                sourceType: nextSource,
                ...(nextSource === "upload"
                  ? { storageProvider: undefined }
                  : { uploadFileName: "", storageProvider: value.storageProvider || DEFAULT_STORAGE_PROVIDER }),
              });
            }}
            className="flex flex-wrap gap-3"
          >
            <div className="flex items-center gap-2 rounded-lg border px-3 py-2">
              <RadioGroupItem value="object-storage" id={`channel-source-object-${value.id}`} />
              <Label htmlFor={`channel-source-object-${value.id}`} className="font-normal">Object storage</Label>
            </div>
            <div className="flex items-center gap-2 rounded-lg border px-3 py-2">
              <RadioGroupItem value="upload" id={`channel-source-upload-${value.id}`} />
              <Label htmlFor={`channel-source-upload-${value.id}`} className="font-normal">Upload file</Label>
            </div>
          </RadioGroup>
        </div>

        {sourceType === "object-storage" ? (
          <div className="grid gap-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="grid gap-2">
                <Label>Provider</Label>
                <Select value={provider} onValueChange={(v) => updateChannel({ storageProvider: v as StorageProvider })}>
                  <SelectTrigger><SelectValue placeholder="Select provider" /></SelectTrigger>
                  <SelectContent>
                    {storageProviders.map((p) => (
                      <SelectItem key={p.id} value={p.id}>{p.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-2">
                <Label>Region (optional)</Label>
                <Input value={value.region || ""} onChange={(e) => updateChannel({ region: e.target.value })} placeholder="us-east-1" />
              </div>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="grid gap-2">
                <Label>Bucket / Container</Label>
                <Input value={value.bucket || ""} onChange={(e) => updateChannel({ bucket: e.target.value })} placeholder="storage://input" />
              </div>
              <div className="grid gap-2">
                <Label>Prefix / Path</Label>
                <Input value={value.prefix || ""} onChange={(e) => updateChannel({ prefix: e.target.value })} placeholder="datasets/default/" />
              </div>
            </div>
            <div className="grid gap-2">
              <Label>Endpoint (optional)</Label>
              <Input value={value.endpoint || ""} onChange={(e) => updateChannel({ endpoint: e.target.value })} placeholder="https://minio.example.com" />
              <p className="text-xs text-slate-500">Leave blank for managed services; specify an endpoint for self-hosted or custom object storage.</p>
            </div>
          </div>
        ) : (
          <div className="grid gap-2">
            <Label>Upload file</Label>
            <input
              ref={fileInputRef}
              type="file"
              onChange={(e) => {
                const file = e.target.files?.[0];
                updateChannel({ uploadFileName: file ? file.name : "" });
              }}
              className="hidden"
            />
            <div className="flex items-center justify-between rounded-md border border-slate-200 bg-white px-3 py-2 shadow-sm">
              <span className={`text-sm truncate ${value.uploadFileName ? "text-slate-700" : "text-slate-400 italic"}`}>
                {value.uploadFileName || "No file chosen"}
              </span>
              <Button
                type="button"
                size="sm"
                variant="secondary"
                className="w-[94px]"
                onClick={() => fileInputRef.current?.click()}
              >
                {value.uploadFileName ? "Change" : "Browse"}
              </Button>
            </div>
          </div>
        )}
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <div className="grid gap-2">
          <Label>Distribution</Label>
          <Select value={value.distribution || "FullyReplicated"} onValueChange={(v) => updateChannel({ distribution: v as Channel["distribution"] })}>
            <SelectTrigger><SelectValue placeholder="Distribution" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="FullyReplicated">Fully replicated</SelectItem>
              <SelectItem value="ShardedByKey">Sharded by key</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="grid gap-2">
          <Label>Record wrapper</Label>
          <Select value={value.recordWrapperType || "None"} onValueChange={(v) => updateChannel({ recordWrapperType: v as Channel["recordWrapperType"] })}>
            <SelectTrigger><SelectValue placeholder="Record wrapper" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="None">None</SelectItem>
              <SelectItem value="RecordIO">RecordIO</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="grid gap-2">
          <Label>Input mode override (optional)</Label>
          <Select
            value={value.inputModeOverride ?? INPUT_MODE_NONE}
            onValueChange={(v) =>
              updateChannel({
                inputModeOverride: v === INPUT_MODE_NONE ? undefined : (v as Channel["inputModeOverride"]),
              })
            }
          >
            <SelectTrigger><SelectValue placeholder="—" /></SelectTrigger>
            <SelectContent>
              <SelectItem value={INPUT_MODE_NONE}>—</SelectItem>
              <SelectItem value="File">File</SelectItem>
              <SelectItem value="Pipe">Pipe</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  );
}

export default function CreateTrainingJobUI() {
  const router = useRouter();
  const [form, setForm] = useState<TrainingJobForm>({
    jobName: generateJobName(),
    priority: 500,
    region: REGION,
    algorithm: {
      source: "builtin",
      algorithmName: builtinAlgorithms[0].id,
      enableMetrics: true,
    },
    resources: {
      instanceResources: { cpuCores: 4, memoryGiB: 16, gpuCount: 0 },
      instanceCount: 1,
      volumeSizeGB: 50,
      distributed: { enabled: false, strategy: "mpi" },
    },
    stoppingCondition: { maxRuntimeSeconds: 3600 * 4 },
    inputDataConfig: [
      {
        id: randomId("channel"),
        channelName: "train",
        sourceType: "object-storage",
        storageProvider: DEFAULT_STORAGE_PROVIDER,
        endpoint: "https://minio.local",
        bucket: "storage://input",
        prefix: "datasets/default/",
        region: REGION,
        contentType: "",
        compressionType: "None",
        distribution: "FullyReplicated",
        recordWrapperType: "None",
      },
    ],
    outputDataConfig: { artifactUri: "storage://output/artifacts/" },
  });

  const [reviewOpen, setReviewOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitResult, setSubmitResult] = useState<null | { ok: boolean; message: string }>(null);

  const payload = useMemo(() => formToPayload(form), [form]);
  const errors = useMemo(() => validateForm(form), [form]);

  function update<K extends keyof TrainingJobForm>(key: K, value: TrainingJobForm[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function updateAlgorithm(partial: Partial<TrainingJobForm["algorithm"]>) {
    update("algorithm", { ...form.algorithm, ...partial });
  }

  function updateResources(partial: Partial<TrainingJobForm["resources"]>) {
    update("resources", { ...form.resources, ...partial });
  }

  function addChannel(template?: Partial<Channel>) {
    const next: Channel = {
      id: randomId("channel"),
      channelName: template?.channelName || "",
      sourceType: template?.sourceType || "object-storage",
      storageProvider: template?.storageProvider || DEFAULT_STORAGE_PROVIDER,
      endpoint: template?.endpoint || "https://minio.local",
      bucket: template?.bucket || "storage://input",
      prefix: template?.prefix || "datasets/default/",
      region: template?.region || REGION,
      uploadFileName: template?.uploadFileName || "",
      contentType: template?.contentType || "",
      compressionType: template?.compressionType || "None",
      distribution: template?.distribution || "FullyReplicated",
      recordWrapperType: template?.recordWrapperType || "None",
      inputModeOverride: template?.inputModeOverride,
    };
    update("inputDataConfig", [...form.inputDataConfig, next]);
  }

  function removeChannel(id: string) {
    update("inputDataConfig", form.inputDataConfig.filter((c) => c.id !== id));
  }

  function duplicateChannel(id: string) {
    const src = form.inputDataConfig.find((c) => c.id === id);
    if (!src) return;
    const copy = { ...deepClone(src), id: randomId("channel"), channelName: `${src.channelName}-copy` };
    update("inputDataConfig", [...form.inputDataConfig, copy]);
  }

  function moveChannel(id: string, dir: -1 | 1) {
    const idx = form.inputDataConfig.findIndex((c) => c.id === id);
    if (idx < 0) return;
    const arr = [...form.inputDataConfig];
    const j = idx + dir;
    if (j < 0 || j >= arr.length) return;
    const [item] = arr.splice(idx, 1);
    arr.splice(j, 0, item);
    update("inputDataConfig", arr);
  }

  async function submit() {
    if (errors.length) {
      setSubmitResult({ ok: false, message: "Fix form errors and try again." });
      return;
    }

    setSubmitting(true);
    try {
      const payload = formToPayload(form);
      const job: StoredJob = {
        id: form.jobName,
        algorithm:
          form.algorithm.source === "builtin"
            ? builtinAlgorithms.find((a) => a.id === form.algorithm.algorithmName)?.name || "Built-in"
            : form.algorithm.imageUri || "Custom Container",
        createdAt: Date.now(),
        priority: form.priority,
        status: "Pending",
      pendingUntil: Date.now() + 15000,
      };

      const persistenceResult = await persistPayloadOnServer(payload, job);
      const serverMessage = persistenceResult.ok
        ? ` JSON saved to tmp/jobs/${persistenceResult.filename ?? "(see server directory)"}.`
        : " Server copy could not be saved (check logs).";
      setSubmitResult({
        ok: true,
        message: `Training job created.${serverMessage}`,
      });
      setReviewOpen(false);
      if (typeof window !== "undefined") {

        const search = window.location.search || "";
        const pathname = window.location.pathname;
        let redirect = LIST_ROUTE;
        if (pathname.startsWith("/_/training-jobs")) {
          redirect = "/_/training-job";
        } else if (pathname.startsWith("/_/training-job")) {
          redirect = "/_/training-job";
        } else if (BASE_PATH && pathname.startsWith(BASE_PATH)) {
          redirect = LIST_ROUTE;
        }

        window.location.assign(`${redirect}${search}`);
      } else {
        router.push(LIST_ROUTE);
      }
    } catch (error) {
      console.error("Failed to submit job", error);
      setSubmitResult({ ok: false, message: "Unexpected error while creating job." });
    } finally {
      setSubmitting(false);
    }
  }

  const maxRuntimeHours = Math.floor(form.stoppingCondition.maxRuntimeSeconds / 3600);
  const maxRuntimeMinutes = Math.floor((form.stoppingCondition.maxRuntimeSeconds % 3600) / 60);

  return (
    <div className="w-full min-h-screen bg-gradient-to-b from-slate-50 to-white">
      <div className="mx-auto max-w-6xl p-6 md:p-10">
        <div className="mb-6">
          <div className="space-y-2">
            <Link href="/training-job" className="text-sm text-blue-600 hover:underline">
              ← Back to job list
            </Link>
            <div>
              <h1 className="text-3xl font-semibold tracking-tight">Create Training Job</h1>
              <p className="text-slate-600 mt-1">Define job settings, algorithm, resources, inputs, stopping conditions, and outputs.</p>
            </div>
          </div>
        </div>

        <div className="grid gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Job settings</CardTitle>
              <CardDescription>Name the training job.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <div className="grid gap-2">
                <Label>Job name</Label>
                <div className="flex gap-2 mt-1.5">
                  <Input value={form.jobName} onChange={(e) => update("jobName", e.target.value)} placeholder="train-2025…" />
                  <Button type="button" variant="secondary" onClick={() => update("jobName", generateJobName("train"))}>
                    Generate
                  </Button>
                </div>
                {!JOB_NAME_REGEX.test(form.jobName || "") && (
                  <p className="text-xs text-red-600 mt-1">Must be lowercase alphanumerics and dashes; max 63 chars; cannot start/end with a dash.</p>
                )}
              </div>
              <div className="grid gap-2">
                <Label>Priority</Label>
                <Input
                  type="number"
                  min={1}
                  max={1000}
                  value={form.priority}
                  onChange={(e) => {
                    const raw = Number(e.target.value);
                    const next = Number.isFinite(raw) ? Math.min(1000, Math.max(1, raw)) : form.priority;
                    update("priority", next);
                  }}
                />
                <p className="text-xs text-slate-500">1 (lowest) — 1000 (highest). Schedulers use this to rank jobs.</p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Algorithm</CardTitle>
              <CardDescription>Select a built-in algorithm or provide a custom container.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <div className="grid gap-2">
                <Label>Source</Label>
                <RadioGroup
                  value={form.algorithm.source}
                  onValueChange={(v) => {
                    const next = v as AlgorithmSource;
                    const defaultBuiltin = form.algorithm.algorithmName || builtinAlgorithms[0].id;
                    updateAlgorithm({
                      source: next,
                      algorithmName: next === "builtin" ? defaultBuiltin : undefined,
                      imageUri: next === "container" ? form.algorithm.imageUri || "" : undefined,
                    });
                  }}
                  className="flex flex-wrap gap-3"
                >
                  <div className="flex items-center gap-2 rounded-lg border px-3 py-2">
                    <RadioGroupItem value="builtin" id="algorithm-source-builtin" />
                    <Label htmlFor="algorithm-source-builtin" className="font-normal">
                      Built-in algorithm
                    </Label>
                  </div>
                  <div className="flex items-center gap-2 rounded-lg border px-3 py-2">
                    <RadioGroupItem value="container" id="algorithm-source-container" />
                    <Label htmlFor="algorithm-source-container" className="font-normal">
                      Custom container
                    </Label>
                  </div>
                </RadioGroup>
              </div>
              {form.algorithm.source === "builtin" ? (
                <div className="grid gap-2">
                  <Label>Built-in algorithm</Label>
                  <Select
                    value={form.algorithm.algorithmName}
                    onValueChange={(value) => updateAlgorithm({ algorithmName: value, imageUri: undefined })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select algorithm" />
                    </SelectTrigger>
                    <SelectContent>
                      {builtinAlgorithms.map((algo) => (
                        <SelectItem key={algo.id} value={algo.id}>
                          {algo.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              ) : (
                <div className="grid gap-2">
                  <Label>Container image URI</Label>
                  <Input
                    value={form.algorithm.imageUri || ""}
                    onChange={(e) => updateAlgorithm({ imageUri: e.target.value, algorithmName: undefined })}
                    placeholder="registry.example.com/ml/training:latest"
                  />
                  <p className="text-xs text-slate-500">Provide a fully qualified container image URI.</p>
                </div>
              )}
              <div className="flex items-center justify-between rounded-lg border border-slate-200 bg-white px-3 py-2">
                <div>
                  <p className="text-sm font-medium text-slate-700">Enable metrics</p>
                  <p className="text-xs text-slate-500">Emit algorithm metrics to the training dashboard.</p>
                </div>
                <Switch checked={form.algorithm.enableMetrics} onCheckedChange={(checked) => updateAlgorithm({ enableMetrics: checked })} />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Resources &amp; scaling</CardTitle>
              <CardDescription>Configure compute resources, storage, and distribution.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <div className="grid gap-4 md:grid-cols-3">
                <div className="grid gap-2">
                  <Label>CPUs per instance</Label>
                  <Input
                    type="number"
                    min={1}
                    value={form.resources.instanceResources.cpuCores}
                    onChange={(e) => {
                      const raw = Number(e.target.value);
                      const next = Number.isFinite(raw) ? Math.max(1, Math.floor(raw)) : form.resources.instanceResources.cpuCores;
                      updateResources({
                        instanceResources: { ...form.resources.instanceResources, cpuCores: next },
                      });
                    }}
                  />
                </div>
                <div className="grid gap-2">
                  <Label>Memory (GiB) per instance</Label>
                  <Input
                    type="number"
                    min={1}
                    value={form.resources.instanceResources.memoryGiB}
                    onChange={(e) => {
                      const raw = Number(e.target.value);
                      const next = Number.isFinite(raw) ? Math.max(1, Math.floor(raw)) : form.resources.instanceResources.memoryGiB;
                      updateResources({
                        instanceResources: { ...form.resources.instanceResources, memoryGiB: next },
                      });
                    }}
                  />
                </div>
                <div className="grid gap-2">
                  <Label>GPUs per instance</Label>
                  <Input
                    type="number"
                    min={0}
                    value={form.resources.instanceResources.gpuCount}
                    onChange={(e) => {
                      const raw = Number(e.target.value);
                      const next = Number.isFinite(raw) ? Math.max(0, Math.floor(raw)) : form.resources.instanceResources.gpuCount;
                      updateResources({
                        instanceResources: { ...form.resources.instanceResources, gpuCount: next },
                      });
                    }}
                  />
                </div>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="grid gap-2">
                  <Label>Instance count</Label>
                  <Input
                    type="number"
                    min={1}
                    value={form.resources.instanceCount}
                    onChange={(e) => {
                      const raw = Number(e.target.value);
                      const next = Number.isFinite(raw) ? Math.max(1, Math.floor(raw)) : form.resources.instanceCount;
                      updateResources({ instanceCount: next });
                    }}
                  />
                </div>
                <div className="grid gap-2">
                  <Label>Volume size (GiB)</Label>
                  <Input
                    type="number"
                    min={1}
                    value={form.resources.volumeSizeGB}
                    onChange={(e) => {
                      const raw = Number(e.target.value);
                      const next = Number.isFinite(raw) ? Math.max(1, Math.floor(raw)) : form.resources.volumeSizeGB;
                      updateResources({ volumeSizeGB: next });
                    }}
                  />
                </div>
              </div>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div>
                    <p className="text-sm font-medium text-slate-700">Distributed training</p>
                    <p className="text-xs text-slate-500">Enable if workers coordinate across nodes.</p>
                  </div>
                  <Switch
                    checked={form.resources.distributed.enabled}
                    onCheckedChange={(checked) =>
                      updateResources({
                        distributed: {
                          ...form.resources.distributed,
                          enabled: checked,
                          ...(checked ? {} : { worldSize: undefined, processesPerHost: undefined }),
                        },
                      })
                    }
                  />
                </div>
                {form.resources.distributed.enabled && (
                  <div className="mt-4 grid gap-4 md:grid-cols-3">
                    <div className="grid gap-2">
                      <Label>Strategy</Label>
                      <Select
                        value={form.resources.distributed.strategy}
                        onValueChange={(value) =>
                          updateResources({
                            distributed: { ...form.resources.distributed, strategy: value as typeof form.resources.distributed.strategy },
                          })
                        }
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select strategy" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="mpi">MPI / Horovod</SelectItem>
                          <SelectItem value="ps">Parameter Server</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="grid gap-2">
                      <Label>World size</Label>
                      <Input
                        type="number"
                        min={1}
                        value={form.resources.distributed.worldSize ?? ""}
                        onChange={(e) => {
                          const raw = Number(e.target.value);
                          const next = Number.isFinite(raw) ? Math.max(1, Math.floor(raw)) : undefined;
                          updateResources({
                            distributed: { ...form.resources.distributed, worldSize: next },
                          });
                        }}
                        placeholder="Total processes"
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label>Processes per host</Label>
                      <Input
                        type="number"
                        min={1}
                        value={form.resources.distributed.processesPerHost ?? ""}
                        onChange={(e) => {
                          const raw = Number(e.target.value);
                          const next = Number.isFinite(raw) ? Math.max(1, Math.floor(raw)) : undefined;
                          updateResources({
                            distributed: { ...form.resources.distributed, processesPerHost: next },
                          });
                        }}
                        placeholder="Optional"
                      />
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Input channels</CardTitle>
              <CardDescription>Connect datasets or uploads for training.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              {form.inputDataConfig.length === 0 && (
                <p className="text-sm text-slate-600">No channels configured yet. Add at least one to continue.</p>
              )}
              {form.inputDataConfig.map((channel, idx) => (
                <div key={channel.id} className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <p className="text-sm font-semibold text-slate-700">{channel.channelName || `Channel ${idx + 1}`}</p>
                      <p className="text-xs text-slate-500">#{idx + 1}</p>
                    </div>
                    <div className="flex items-center gap-1">
                      <Badge variant="outline">
                        {channel.sourceType === "upload"
                          ? "Upload"
                          : storageProviderLabels[channel.storageProvider || DEFAULT_STORAGE_PROVIDER]}
                      </Badge>
                      <Button
                        type="button"
                        size="icon"
                        variant="ghost"
                        onClick={() => moveChannel(channel.id, -1)}
                        disabled={idx === 0}
                      >
                        <ChevronUp className="h-4 w-4" />
                        <span className="sr-only">Move up</span>
                      </Button>
                      <Button
                        type="button"
                        size="icon"
                        variant="ghost"
                        onClick={() => moveChannel(channel.id, 1)}
                        disabled={idx === form.inputDataConfig.length - 1}
                      >
                        <ChevronDown className="h-4 w-4" />
                        <span className="sr-only">Move down</span>
                      </Button>
                      <Button type="button" size="icon" variant="ghost" onClick={() => duplicateChannel(channel.id)}>
                        <Copy className="h-4 w-4" />
                        <span className="sr-only">Duplicate channel</span>
                      </Button>
                      <Button
                        type="button"
                        size="icon"
                        variant="ghost"
                        onClick={() => removeChannel(channel.id)}
                        disabled={form.inputDataConfig.length === 1}
                      >
                        <Trash2 className="h-4 w-4 text-red-600" />
                        <span className="sr-only">Remove channel</span>
                      </Button>
                    </div>
                  </div>
                  <div className="mt-4">
                    <ChannelEditor
                      value={channel}
                      onChange={(next) => {
                        const nextChannels = [...form.inputDataConfig];
                        nextChannels[idx] = next;
                        update("inputDataConfig", nextChannels);
                      }}
                    />
                  </div>
                </div>
              ))}
              <div className="flex flex-wrap gap-2">
                <Button type="button" onClick={() => addChannel({ channelName: `channel-${form.inputDataConfig.length + 1}` })}>
                  <Plus className="mr-2 h-4 w-4" /> Add channel
                </Button>
                <Button type="button" variant="outline" onClick={() => addChannel({ channelName: "validation" })}>
                  Quick add validation
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Stopping condition</CardTitle>
              <CardDescription>Limit how long the training job should run.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2">
              <div className="grid gap-2">
                <Label>Hours</Label>
                <Input
                  type="number"
                  min={0}
                  value={maxRuntimeHours}
                  onChange={(e) => {
                    const raw = Number(e.target.value);
                    const next = Number.isFinite(raw) ? Math.max(0, Math.floor(raw)) : maxRuntimeHours;
                    const total = secondsFromHM(next, maxRuntimeMinutes);
                    update("stoppingCondition", { maxRuntimeSeconds: total });
                  }}
                />
              </div>
              <div className="grid gap-2">
                <Label>Minutes</Label>
                <Input
                  type="number"
                  min={0}
                  max={59}
                  value={maxRuntimeMinutes}
                  onChange={(e) => {
                    const raw = Number(e.target.value);
                    const next = Number.isFinite(raw) ? Math.max(0, Math.min(59, Math.floor(raw))) : maxRuntimeMinutes;
                    const total = secondsFromHM(maxRuntimeHours, next);
                    update("stoppingCondition", { maxRuntimeSeconds: total });
                  }}
                />
              </div>
              <p className="text-xs text-slate-500 md:col-span-2">
                Total runtime limit: {Math.max(0, maxRuntimeHours)}h {Math.max(0, maxRuntimeMinutes)}m.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Output data</CardTitle>
              <CardDescription>Where to store model artifacts and logs.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-2">
              <Label>Artifact URI</Label>
              <Input
                value={form.outputDataConfig.artifactUri}
                onChange={(e) => update("outputDataConfig", { artifactUri: e.target.value })}
                placeholder="storage://output/artifacts/"
              />
              <p className="text-xs text-slate-500">Provide a URI for an object store or shared filesystem where the job can write results.</p>
            </CardContent>
          </Card>
        </div>

        <div className="mt-10 flex flex-col gap-4 rounded-2xl border border-slate-200 bg-white/80 p-4 shadow-sm md:flex-row md:items-center md:justify-between">
          <div className="text-sm text-slate-600">
            {errors.length > 0 ? (
              <>
                <span className="font-semibold text-red-600">
                  {errors.length} error{errors.length === 1 ? "" : "s"}
                </span>{" "}
                to resolve before submitting.
              </>
            ) : (
              <>
                <span className="font-semibold text-emerald-600">Ready to submit.</span>{" "}
                Resources:&nbsp;
                <span className="font-medium text-slate-700">{form.resources.instanceResources.cpuCores}</span> CPU /{" "}
                <span className="font-medium text-slate-700">{form.resources.instanceResources.memoryGiB}</span> GiB /{" "}
                <span className="font-medium text-slate-700">{form.resources.instanceResources.gpuCount}</span> GPU · Instances:{" "}
                <span className="font-medium text-slate-700">{form.resources.instanceCount}</span> · Volume:{" "}
                <span className="font-medium text-slate-700">{form.resources.volumeSizeGB}</span> GiB
              </>
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            <Dialog open={reviewOpen} onOpenChange={setReviewOpen}>
              <DialogTrigger asChild>
                <Button disabled={errors.length > 0}>
                  <FileJson2 className="mr-2 h-4 w-4" /> Review &amp; Submit
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-3xl">
                <DialogHeader>
                  <DialogTitle>Review configuration</DialogTitle>
                </DialogHeader>
                <div className="grid gap-4">
                  {errors.length > 0 && (
                    <Card className="border-red-200 bg-red-50">
                      <CardHeader className="py-3">
                        <CardTitle className="text-red-700 text-base">Please resolve the following:</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ul className="list-disc ml-5 space-y-1 text-red-700 text-sm">
                          {errors.map((e, i) => (
                            <li key={i}>{e}</li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>
                  )}
                  <div className="rounded-xl border bg-slate-50 p-4">
                    <pre className="text-xs overflow-auto max-h-64">
{JSON.stringify(payload, null, 2)}
                    </pre>
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button variant="secondary" onClick={() => navigator.clipboard.writeText(JSON.stringify(payload, null, 2))}>
                      <Copy className="mr-2 h-4 w-4" /> Copy JSON
                    </Button>
                    <Button onClick={submit} disabled={submitting}>
                      {submitting ? "Submitting…" : "Submit"}
                    </Button>
                  </div>
                  {submitResult && (
                    <p className={`text-sm ${submitResult.ok ? "text-emerald-600" : "text-red-600"}`}>{submitResult.message}</p>
                  )}
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </div>
      </div>
    </div>
  );
}
