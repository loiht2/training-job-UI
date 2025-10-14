'use client';

import React, { useMemo, useRef, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChevronDown, ChevronUp, Copy, Download, FileJson2, Plus, Trash2 } from "lucide-react";

// ---- Types ----
type AlgorithmSource = "builtin" | "container";

type Channel = {
  id: string;
  channelName: string; // train/validation/test/...
  s3DataType: "S3Prefix" | "S3Object";
  s3Uri: string;
  sourceType?: "s3" | "upload";
  uploadFileName?: string;
  contentType?: string;
  compressionType?: "None" | "Gzip";
  distribution?: "FullyReplicated" | "ShardedByS3Key";
  recordWrapperType?: "None" | "RecordIO";
  inputModeOverride?: "File" | "Pipe";
};

type TrainingJobForm = {
  jobName: string;
  region: string;
  algorithm: {
    source: AlgorithmSource;
    algorithmName?: string; // builtin
    imageUri?: string; // container
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
  outputDataConfig: { s3OutputPath: string };
};

// ---- Constants / Mock Data ----
const builtinAlgorithms = [
  { id: "xgboost", name: "XGBoost (1.7)" },
  { id: "lightgbm", name: "LightGBM" },
  { id: "image-classification", name: "Image Classification (MXNet)" },
  { id: "object-detection", name: "Object Detection (SSD)" },
];

const REGION = "us-east-1"; // demo

// ---- Utils ----
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

function deepClone<T>(x: T): T { return JSON.parse(JSON.stringify(x)); }

function formToPayload(form: TrainingJobForm) {
  return {
    jobName: form.jobName,
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

function validateForm(form: TrainingJobForm) {
  const errors: string[] = [];
  if (!form.jobName || !JOB_NAME_REGEX.test(form.jobName)) {
    errors.push("Job name is required and must match ^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$");
  }
  const alg = form.algorithm;
  if (alg.source === "builtin" && !alg.algorithmName) errors.push("Select a built-in algorithm.");
  if (alg.source === "container" && !alg.imageUri) errors.push("Container image URI is required.");
  if (!form.inputDataConfig?.length) errors.push("At least one input channel is required.");
  const hasTrain = form.inputDataConfig.some((c) => c.channelName === "train");
  if (!hasTrain) errors.push("A 'train' channel is required.");
  form.inputDataConfig.forEach((c, idx) => {
    const sourceType = c.sourceType || "s3";
    if (!c.channelName) errors.push(`Channel #${idx + 1}: name is required.`);
    if (sourceType === "s3") {
      if (!c.s3Uri) errors.push(`Channel '${c.channelName || idx + 1}': S3 location is required.`);
    } else {
      if (!c.uploadFileName) errors.push(`Channel '${c.channelName || idx + 1}': upload file is required.`);
    }
  });
  if (!form.outputDataConfig.s3OutputPath) errors.push("Output S3 path is required.");
  return errors;
}

function downloadJSON(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}

// ---- Channel Editor Component ----
function ChannelEditor({ value, onChange }: { value: Channel; onChange: (c: Channel) => void }) {
  const sourceType = value.sourceType || "s3";
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

      <div className="grid gap-3">
        <div className="grid gap-2">
          <Label>Data source</Label>
          <RadioGroup
            value={sourceType}
            onValueChange={(v) => {
              const nextSource = v as Channel["sourceType"];
              updateChannel({
                sourceType: nextSource,
                ...(nextSource === "upload" ? {} : { uploadFileName: "" }),
              });
            }}
            className="flex flex-wrap gap-3"
          >
            <div className="flex items-center gap-2 rounded-lg border px-3 py-2">
              <RadioGroupItem value="s3" id={`channel-source-s3-${value.id}`} />
              <Label htmlFor={`channel-source-s3-${value.id}`} className="font-normal">S3</Label>
            </div>
            <div className="flex items-center gap-2 rounded-lg border px-3 py-2">
              <RadioGroupItem value="upload" id={`channel-source-upload-${value.id}`} />
              <Label htmlFor={`channel-source-upload-${value.id}`} className="font-normal">Upload file</Label>
            </div>
          </RadioGroup>
        </div>

        {sourceType === "s3" ? (
          <>
            <div className="grid gap-2">
              <Label>Location</Label>
              <Input value={value.s3Uri} onChange={(e) => updateChannel({ s3Uri: e.target.value })} placeholder="s3://bucket/prefix" />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="grid gap-2">
                <Label>Data type</Label>
                <Select value={value.s3DataType} onValueChange={(v) => updateChannel({ s3DataType: v as Channel["s3DataType"] })}>
                  <SelectTrigger><SelectValue placeholder="S3 data type" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="S3Prefix">S3Prefix</SelectItem>
                    <SelectItem value="S3Object">S3Object</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-2">
                <Label>Distribution</Label>
                <Select value={value.distribution || "FullyReplicated"} onValueChange={(v) => updateChannel({ distribution: v as Channel["distribution"] })}>
                  <SelectTrigger><SelectValue placeholder="Distribution" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="FullyReplicated">FullyReplicated</SelectItem>
                    <SelectItem value="ShardedByS3Key">ShardedByS3Key</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </>
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
                {value.uploadFileName || "No file choosen"}
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
    </div>
  );
}

// ---- Main UI ----
export default function CreateTrainingJobUI() {
  const [form, setForm] = useState<TrainingJobForm>({
    jobName: generateJobName(),
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
        s3DataType: "S3Prefix",
        s3Uri: "s3://my-bucket/train/",
        sourceType: "s3",
        uploadFileName: "",
        contentType: "",
        compressionType: "None",
        distribution: "FullyReplicated",
        recordWrapperType: "None",
      },
    ],
    outputDataConfig: { s3OutputPath: "s3://my-bucket/output/" },
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
      s3DataType: template?.s3DataType || "S3Prefix",
      s3Uri: template?.s3Uri || "",
      sourceType: template?.sourceType || "s3",
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

  function submit() {
    setSubmitting(true);
    // Simulate API call
    setTimeout(() => {
      setSubmitting(false);
      setSubmitResult({ ok: errors.length === 0, message: errors.length ? "Fix form errors and try again." : "Training job created successfully." });
      if (errors.length === 0) setReviewOpen(false);
    }, 600);
  }

  const maxRuntimeHours = Math.floor(form.stoppingCondition.maxRuntimeSeconds / 3600);
  const maxRuntimeMinutes = Math.floor((form.stoppingCondition.maxRuntimeSeconds % 3600) / 60);

  return (
    <div className="w-full min-h-screen bg-gradient-to-b from-slate-50 to-white">
      <div className="mx-auto max-w-6xl p-6 md:p-10">
        <div className="mb-6">
          <h1 className="text-3xl font-semibold tracking-tight">Create Training Job</h1>
          <p className="text-slate-600 mt-1">Define job settings, algorithm, resources, inputs, stopping conditions, and outputs.</p>
        </div>

        <div className="grid gap-6">
          {/* Job Settings */}
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
                  <Button type="button" variant="secondary" onClick={() => update("jobName", generateJobName("train"))}>Generate</Button>
                </div>
                {!JOB_NAME_REGEX.test(form.jobName || "") && (
                  <p className="text-xs text-red-600 mt-1">Must be lowercase alphanumerics and dashes; max 63 chars; cannot start/end with a dash.</p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Algorithm */}
          <Card>
            <CardHeader>
              <CardTitle>Algorithm</CardTitle>
              <CardDescription>Choose a built-in algorithm, or your container image.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label>Source</Label>
                  <RadioGroup
                    value={form.algorithm.source}
                    onValueChange={(v: AlgorithmSource) => {
                      const next: Partial<TrainingJobForm["algorithm"]> = { source: v };
                      if (v === "builtin") {
                        next.algorithmName = builtinAlgorithms[0].id;
                        delete next.imageUri;
                      } else if (v === "container") {
                        next.imageUri = "";
                        delete next.algorithmName;
                      }
                      update("algorithm", { ...form.algorithm, ...next });
                    }}
                    className="grid grid-cols-2 gap-2 mt-2"
                  >
                    <div className="flex items-center space-x-2 rounded-xl border p-2">
                      <RadioGroupItem value="builtin" id="src-builtin" />
                      <Label htmlFor="src-builtin">Built-in</Label>
                    </div>
                    <div className="flex items-center space-x-2 rounded-xl border p-2">
                      <RadioGroupItem value="container" id="src-container" />
                      <Label htmlFor="src-container">Custom</Label>
                    </div>
                  </RadioGroup>
                </div>
                <div className="grid gap-4">
                  {form.algorithm.source === "builtin" && (
                    <div className="grid gap-2">
                      <Label>Built-in algorithm</Label>
                      <Select value={form.algorithm.algorithmName} onValueChange={(v) => updateAlgorithm({ algorithmName: v })}>
                        <SelectTrigger><SelectValue placeholder="Select algorithm" /></SelectTrigger>
                        <SelectContent>
                          {builtinAlgorithms.map((a) => (
                            <SelectItem key={a.id} value={a.id}>{a.name}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                  {form.algorithm.source === "container" && (
                    <div className="grid gap-2">
                      <Label>Image URI</Label>
                      <Input value={form.algorithm.imageUri || ""} onChange={(e) => updateAlgorithm({ imageUri: e.target.value })} placeholder="docker.io/dcnlab/training-image:latest" />
                    </div>
                  )}
                </div>
              </div>

              <div className="grid gap-2">
                <Label className="flex items-center gap-2">Enable metrics <Badge variant="secondary">recommended</Badge></Label>
                <div className="flex items-center justify-between rounded-xl border p-3 mt-1.5">
                  <p className="text-sm text-slate-600">Publish training metrics to your observability backend.</p>
                  <Switch checked={form.algorithm.enableMetrics} onCheckedChange={(v) => updateAlgorithm({ enableMetrics: !!v })} />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Resources */}
          <Card>
            <CardHeader className="space-y-2">
              <CardTitle>Resource configuration</CardTitle>
              <CardDescription>Define instance resources, count, volume, and optionally enable distributed training.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-5">
              <div className="grid gap-4">
                <div className="grid gap-2">
                  <Label>Instance resource</Label>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-2">
                    <div className="grid gap-2">
                      <Label className="text-xs text-slate-500">CPU (vCPU)</Label>
                      <Input
                        type="number"
                        min={0}
                        value={form.resources.instanceResources.cpuCores}
                        onChange={(e) =>
                          updateResources({
                            instanceResources: {
                              ...form.resources.instanceResources,
                              cpuCores: Math.max(0, Number(e.target.value) || 0),
                            },
                          })
                        }
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label className="text-xs text-slate-500">Memory (GB)</Label>
                      <Input
                        type="number"
                        min={0}
                        value={form.resources.instanceResources.memoryGiB}
                        onChange={(e) =>
                          updateResources({
                            instanceResources: {
                              ...form.resources.instanceResources,
                              memoryGiB: Math.max(0, Number(e.target.value) || 0),
                            },
                          })
                        }
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label className="text-xs text-slate-500">GPU count</Label>
                      <Input
                        type="number"
                        min={0}
                        value={form.resources.instanceResources.gpuCount}
                        onChange={(e) =>
                          updateResources({
                            instanceResources: {
                              ...form.resources.instanceResources,
                              gpuCount: Math.max(0, Number(e.target.value) || 0),
                            },
                          })
                        }
                      />
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
                <div className="grid gap-2">
                  <Label>Instance count</Label>
                  <div className="flex items-center gap-2">
                    <Button size="icon" variant="secondary" onClick={() => updateResources({ instanceCount: Math.max(1, form.resources.instanceCount - 1) })}>-</Button>
                    <Input
                      type="number"
                      value={form.resources.instanceCount}
                      onChange={(e) => updateResources({ instanceCount: Math.max(1, Number(e.target.value) || 1) })}
                      className="w-32 text-center"
                    />
                    <Button size="icon" variant="secondary" onClick={() => updateResources({ instanceCount: form.resources.instanceCount + 1 })}>+</Button>
                  </div>
                </div>
                <div className="grid gap-2">
                  <Label>Volume size (GB)</Label>
                  <Input
                    type="number"
                    min={0}
                    value={form.resources.volumeSizeGB}
                    onChange={(e) => updateResources({ volumeSizeGB: Math.max(0, Number(e.target.value) || 0) })}
                  />
                </div>
              </div>
              </div>

              <Accordion type="single" collapsible className="rounded-xl border w-full">
                  <AccordionItem value="advanced">
                    <AccordionTrigger className="px-4">Advanced: Distributed training</AccordionTrigger>
                    <AccordionContent className="px-4 pb-4">
                      <div className="flex items-center justify-between">
                        <Label>Enable distributed</Label>
                        <Switch checked={form.resources.distributed.enabled} onCheckedChange={(v) => updateResources({ distributed: { ...form.resources.distributed, enabled: !!v } })} />
                      </div>
                      {form.resources.distributed.enabled && (
                        <div className="grid grid-cols-2 gap-4 mt-3">
                          <div className="grid gap-2">
                            <Label>Strategy</Label>
                            <Select value={form.resources.distributed.strategy} onValueChange={(v) => updateResources({ distributed: { ...form.resources.distributed, strategy: v as "mpi" | "ps" } })}>
                              <SelectTrigger><SelectValue /></SelectTrigger>
                              <SelectContent>
                                <SelectItem value="mpi">MPI / NCCL</SelectItem>
                                <SelectItem value="ps">Parameter Server</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="grid gap-2">
                            <Label>World size</Label>
                            <Input type="number" value={form.resources.distributed.worldSize || form.resources.instanceCount} onChange={(e) => updateResources({ distributed: { ...form.resources.distributed, worldSize: Math.max(1, Number(e.target.value) || 1) } })} />
                          </div>
                          <div className="grid gap-2">
                            <Label>Processes per host</Label>
                            <Input type="number" value={form.resources.distributed.processesPerHost || 1} onChange={(e) => updateResources({ distributed: { ...form.resources.distributed, processesPerHost: Math.max(1, Number(e.target.value) || 1) } })} />
                          </div>
                        </div>
                      )}
                    </AccordionContent>
                  </AccordionItem>
              </Accordion>
            </CardContent>
          </Card>

          {/* Stopping Conditions */}
          <Card>
            <CardHeader>
              <CardTitle>Stopping conditions</CardTitle>
              <CardDescription>Set a hard limit for training runtime.</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
              <div className="grid gap-2">
                <Label>Hours</Label>
                <Input type="number" min={0} value={maxRuntimeHours} onChange={(e) => update("stoppingCondition", { maxRuntimeSeconds: secondsFromHM(Number(e.target.value) || 0, maxRuntimeMinutes) })} />
              </div>
              <div className="grid gap-2">
                <Label>Minutes</Label>
                <Input type="number" min={0} max={59} value={maxRuntimeMinutes} onChange={(e) => update("stoppingCondition", { maxRuntimeSeconds: secondsFromHM(maxRuntimeHours, Math.min(59, Number(e.target.value) || 0)) })} />
              </div>
              <div>
                <div className="text-xs text-slate-600">Total: {Math.floor(form.stoppingCondition.maxRuntimeSeconds / 3600)}h {Math.floor((form.stoppingCondition.maxRuntimeSeconds % 3600) / 60)}m</div>
              </div>
            </CardContent>
          </Card>

          {/* Input Data (Channels) */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div className="space-y-2">
                <CardTitle>Input data configuration</CardTitle>
                <CardDescription>Specify the channels that provide datasets for training and evaluation. At least <span className="font-medium">one</span> channel is required.</CardDescription>
              </div>
              <Button onClick={() => addChannel({ channelName: "", s3Uri: "" })}><Plus className="mr-2 h-4 w-4" /> Add channel</Button>
            </CardHeader>
            <CardContent className="grid gap-4">
              <div className="rounded-xl border overflow-hidden">
                <div className="grid grid-cols-[1.25fr_2.5fr_1fr_1fr_auto] items-center bg-slate-50 px-4 py-2 text-xs font-medium uppercase tracking-wide text-slate-600">
                  <div>Name</div>
                  <div>Location</div>
                  <div>Type</div>
                  <div>Compression</div>
                  <div className="text-right">Actions</div>
                </div>
                {form.inputDataConfig.length ? (
                  <div className="divide-y">
                    {form.inputDataConfig.map((c) => (
                      <ChannelRow
                        key={c.id}
                        c={c}
                        onChange={(nc) => {
                          const arr = form.inputDataConfig.map((x) => (x.id === c.id ? nc : x));
                          update("inputDataConfig", arr);
                        }}
                        onDuplicate={() => duplicateChannel(c.id)}
                        onRemove={() => removeChannel(c.id)}
                        onMoveUp={() => moveChannel(c.id, -1)}
                        onMoveDown={() => moveChannel(c.id, 1)}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="px-4 py-6 text-sm text-slate-500">No channels added yet.</div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Output Artifacts */}
          <Card>
            <CardHeader>
              <CardTitle>Output artifacts</CardTitle>
              <CardDescription>Specify where to store model artifacts.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-2">
              <Label>S3 output path</Label>
              <Input value={form.outputDataConfig.s3OutputPath} onChange={(e) => update("outputDataConfig", { ...form.outputDataConfig, s3OutputPath: e.target.value })} placeholder="s3://bucket/output/" />
            </CardContent>
          </Card>
        </div>

        {/* Sticky Submit Bar */}
        <div className="sticky bottom-4 mt-8">
          <div className="mx-auto max-w-6xl">
            <div className="rounded-2xl border bg-white/80 backdrop-blur p-3 shadow-xl">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
                <div className="text-sm text-slate-600">
                  {errors.length ? (
                    <span className="text-red-600 font-medium">{errors.length} error(s)</span>
                  ) : (
                    <span className="text-emerald-700 font-medium">Ready to submit</span>
                  )} · Resources:{" "}
                  <span className="font-medium">{form.resources.instanceResources.cpuCores}</span> vCPU /{" "}
                  <span className="font-medium">{form.resources.instanceResources.memoryGiB}</span> GB /{" "}
                  <span className="font-medium">{form.resources.instanceResources.gpuCount}</span> GPU · Count: {form.resources.instanceCount} · Volume: {form.resources.volumeSizeGB} GB
                </div>
                <div className="flex gap-2">
                  <Button variant="secondary" onClick={() => downloadJSON(`${form.jobName || "training-job"}.draft.json`, payload)}>
                    <Download className="mr-2 h-4 w-4" /> Save draft
                  </Button>
                  <Dialog open={reviewOpen} onOpenChange={setReviewOpen}>
                    <DialogTrigger asChild>
                      <Button disabled={errors.length > 0}><FileJson2 className="mr-2 h-4 w-4" /> Review & Submit</Button>
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
        </div>
      </div>
    </div>
  );
}

function ChannelRow({ c, onChange, onDuplicate, onRemove, onMoveUp, onMoveDown }: {
  c: Channel;
  onChange: (c: Channel) => void;
  onDuplicate: () => void;
  onRemove: () => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
}) {
  const [open, setOpen] = useState(false);
  const sourceType = c.sourceType || "s3";
  const location = sourceType === "upload" ? c.uploadFileName || "" : c.s3Uri;
  const typeLabel = sourceType === "upload" ? "Local" : c.s3DataType;

  return (
    <div className="grid grid-cols-[1.25fr_2.5fr_1fr_1fr_auto] items-center px-4 py-3 hover:bg-slate-50">
      <div className="font-medium truncate">{c.channelName || <span className="text-slate-400">(unnamed)</span>}</div>
      <div className="truncate text-slate-700">{location || <span className="text-slate-400">—</span>}</div>
      <div className="text-slate-600">{typeLabel}</div>
      <div className="text-slate-600">{c.compressionType || "None"}</div>
      <div className="flex justify-end gap-2">
        <Sheet open={open} onOpenChange={setOpen}>
          <SheetTrigger asChild>
            <Button size="sm" variant="secondary">Edit</Button>
          </SheetTrigger>
          <SheetContent side="right" className="w-[560px] sm:w-[620px] px-6 py-6">
            <SheetHeader>
              <SheetTitle>Edit channel: {c.channelName || "(unnamed)"}</SheetTitle>
            </SheetHeader>
            <ScrollArea className="h-[calc(100vh-10rem)]">
              <ChannelEditor value={c} onChange={onChange} />
            </ScrollArea>
          </SheetContent>
        </Sheet>
        <Button size="icon" variant="secondary" onClick={onMoveUp} title="Move up"><ChevronUp className="h-4 w-4" /></Button>
        <Button size="icon" variant="secondary" onClick={onMoveDown} title="Move down"><ChevronDown className="h-4 w-4" /></Button>
        <Button size="sm" variant="secondary" onClick={onDuplicate}>Duplicate</Button>
        <Button size="icon" variant="destructive" onClick={onRemove} title="Remove"><Trash2 className="h-4 w-4" /></Button>
      </div>
    </div>
  );
}
