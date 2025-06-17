// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cmd

import (
	"context"
	_ "embed"
	"fmt"
	"io"
	"os"
	"os/signal"
	"regexp"
	"runtime"
	"strings"
	"syscall"
	"time"

	yaml "github.com/goccy/go-yaml"
	"github.com/googleapis/genai-toolbox/internal/log"
	"github.com/googleapis/genai-toolbox/internal/prebuiltconfigs"
	"github.com/googleapis/genai-toolbox/internal/server"
	"github.com/googleapis/genai-toolbox/internal/telemetry"
	"github.com/googleapis/genai-toolbox/internal/util"

	// Import tool packages for side effect of registration
	_ "github.com/googleapis/genai-toolbox/internal/tools/alloydbainl"
	_ "github.com/googleapis/genai-toolbox/internal/tools/bigquery"
	_ "github.com/googleapis/genai-toolbox/internal/tools/bigqueryexecutesql"
	_ "github.com/googleapis/genai-toolbox/internal/tools/bigquerygetdatasetinfo"
	_ "github.com/googleapis/genai-toolbox/internal/tools/bigquerygettableinfo"
	_ "github.com/googleapis/genai-toolbox/internal/tools/bigquerylistdatasetids"
	_ "github.com/googleapis/genai-toolbox/internal/tools/bigquerylisttableids"
	_ "github.com/googleapis/genai-toolbox/internal/tools/bigtable"
	_ "github.com/googleapis/genai-toolbox/internal/tools/couchbase"
	_ "github.com/googleapis/genai-toolbox/internal/tools/dgraph"
	_ "github.com/googleapis/genai-toolbox/internal/tools/http"
	_ "github.com/googleapis/genai-toolbox/internal/tools/mssqlexecutesql"
	_ "github.com/googleapis/genai-toolbox/internal/tools/mssqlsql"
	_ "github.com/googleapis/genai-toolbox/internal/tools/mysqlexecutesql"
	_ "github.com/googleapis/genai-toolbox/internal/tools/mysqlsql"
	_ "github.com/googleapis/genai-toolbox/internal/tools/neo4j"
	_ "github.com/googleapis/genai-toolbox/internal/tools/postgresexecutesql"
	_ "github.com/googleapis/genai-toolbox/internal/tools/postgressql"
	_ "github.com/googleapis/genai-toolbox/internal/tools/redis"
	_ "github.com/googleapis/genai-toolbox/internal/tools/spanner"
	_ "github.com/googleapis/genai-toolbox/internal/tools/spannerexecutesql"
	_ "github.com/googleapis/genai-toolbox/internal/tools/sqlitesql"
	_ "github.com/googleapis/genai-toolbox/internal/tools/valkey"

	"github.com/spf13/cobra"

	_ "github.com/googleapis/genai-toolbox/internal/sources/alloydbpg"
	_ "github.com/googleapis/genai-toolbox/internal/sources/bigquery"
	_ "github.com/googleapis/genai-toolbox/internal/sources/bigtable"
	_ "github.com/googleapis/genai-toolbox/internal/sources/cloudsqlmssql"
	_ "github.com/googleapis/genai-toolbox/internal/sources/cloudsqlmysql"
	_ "github.com/googleapis/genai-toolbox/internal/sources/cloudsqlpg"
	_ "github.com/googleapis/genai-toolbox/internal/sources/couchbase"
	_ "github.com/googleapis/genai-toolbox/internal/sources/dgraph"
	_ "github.com/googleapis/genai-toolbox/internal/sources/http"
	_ "github.com/googleapis/genai-toolbox/internal/sources/mssql"
	_ "github.com/googleapis/genai-toolbox/internal/sources/mysql"
	_ "github.com/googleapis/genai-toolbox/internal/sources/neo4j"
	_ "github.com/googleapis/genai-toolbox/internal/sources/postgres"
	_ "github.com/googleapis/genai-toolbox/internal/sources/redis"
	_ "github.com/googleapis/genai-toolbox/internal/sources/spanner"
	_ "github.com/googleapis/genai-toolbox/internal/sources/sqlite"
	_ "github.com/googleapis/genai-toolbox/internal/sources/valkey"
)

var (
	// versionString stores the full semantic version, including build metadata.
	versionString string
	// versionNum indicates the numerical part fo the version
	//go:embed version.txt
	versionNum string
	// metadataString indicates additional build or distribution metadata.
	buildType string = "dev" // should be one of "dev", "binary", or "container"
	// commitSha is the git commit it was built from
	commitSha string
)

func init() {
	versionString = semanticVersion()
}

// semanticVersion returns the version of the CLI including a compile-time metadata.
func semanticVersion() string {
	metadataStrings := []string{buildType, runtime.GOOS, runtime.GOARCH}
	if commitSha != "" {
		metadataStrings = append(metadataStrings, commitSha)
	}
	v := strings.TrimSpace(versionNum) + "+" + strings.Join(metadataStrings, ".")
	return v
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	if err := NewCommand().Execute(); err != nil {
		exit := 1
		os.Exit(exit)
	}
}

// Command represents an invocation of the CLI.
type Command struct {
	*cobra.Command

	cfg            server.ServerConfig
	logger         log.Logger
	tools_file     string
	prebuiltConfig string
	inStream       io.Reader
	outStream      io.Writer
	errStream      io.Writer
}

// NewCommand returns a Command object representing an invocation of the CLI.
func NewCommand(opts ...Option) *Command {
	in := os.Stdin
	out := os.Stdout
	err := os.Stderr

	baseCmd := &cobra.Command{
		Use:           "toolbox",
		Version:       versionString,
		SilenceErrors: true,
	}
	cmd := &Command{
		Command:   baseCmd,
		inStream:  in,
		outStream: out,
		errStream: err,
	}

	for _, o := range opts {
		o(cmd)
	}

	// Set server version
	cmd.cfg.Version = versionString

	// set baseCmd in, out and err the same as cmd.
	baseCmd.SetIn(cmd.inStream)
	baseCmd.SetOut(cmd.outStream)
	baseCmd.SetErr(cmd.errStream)

	flags := cmd.Flags()
	flags.StringVarP(&cmd.cfg.Address, "address", "a", "127.0.0.1", "Address of the interface the server will listen on.")
	flags.IntVarP(&cmd.cfg.Port, "port", "p", 5000, "Port the server will listen on.")

	flags.StringVar(&cmd.tools_file, "tools_file", "", "File path specifying the tool configuration. Cannot be used with --prebuilt.")
	// deprecate tools_file
	_ = flags.MarkDeprecated("tools_file", "please use --tools-file instead")
	flags.StringVar(&cmd.tools_file, "tools-file", "", "File path specifying the tool configuration. Cannot be used with --prebuilt.")
	flags.Var(&cmd.cfg.LogLevel, "log-level", "Specify the minimum level logged. Allowed: 'DEBUG', 'INFO', 'WARN', 'ERROR'.")
	flags.Var(&cmd.cfg.LoggingFormat, "logging-format", "Specify logging format to use. Allowed: 'standard' or 'JSON'.")
	flags.BoolVar(&cmd.cfg.TelemetryGCP, "telemetry-gcp", false, "Enable exporting directly to Google Cloud Monitoring.")
	flags.StringVar(&cmd.cfg.TelemetryOTLP, "telemetry-otlp", "", "Enable exporting using OpenTelemetry Protocol (OTLP) to the specified endpoint (e.g. 'http://127.0.0.1:4318')")
	flags.StringVar(&cmd.cfg.TelemetryServiceName, "telemetry-service-name", "toolbox", "Sets the value of the service.name resource attribute for telemetry data.")
	flags.StringVar(&cmd.prebuiltConfig, "prebuilt", "", "Use a prebuilt tool configuration by source type. Cannot be used with --tools-file. Allowed: 'alloydb-postgres', 'bigquery', 'cloud-sql-mysql', 'cloud-sql-postgres', 'cloud-sql-mssql', 'postgres', 'spanner', 'spanner-postgres'.")
	flags.BoolVar(&cmd.cfg.Stdio, "stdio", false, "Listens via MCP STDIO instead of acting as a remote HTTP server.")

	// wrap RunE command so that we have access to original Command object
	cmd.RunE = func(*cobra.Command, []string) error { return run(cmd) }

	return cmd
}

type ToolsFile struct {
	Sources      server.SourceConfigs      `yaml:"sources"`
	AuthSources  server.AuthServiceConfigs `yaml:"authSources"` // Deprecated: Kept for compatibility.
	AuthServices server.AuthServiceConfigs `yaml:"authServices"`
	Tools        server.ToolConfigs        `yaml:"tools"`
	Toolsets     server.ToolsetConfigs     `yaml:"toolsets"`
}

// parseEnv replaces environment variables ${ENV_NAME} with their values.
func parseEnv(input string) string {
	re := regexp.MustCompile(`\$\{(\w+)\}`)

	return re.ReplaceAllStringFunc(input, func(match string) string {
		parts := re.FindStringSubmatch(match)
		if len(parts) < 2 {
			// technically shouldn't happen
			return match
		}

		// extract the variable name
		variableName := parts[1]
		if value, found := os.LookupEnv(variableName); found {
			return value
		}
		return match
	})
}

// parseToolsFile parses the provided yaml into appropriate configs.
func parseToolsFile(ctx context.Context, raw []byte) (ToolsFile, error) {
	var toolsFile ToolsFile
	// Replace environment variables if found
	raw = []byte(parseEnv(string(raw)))
	// Parse contents
	err := yaml.UnmarshalContext(ctx, raw, &toolsFile, yaml.Strict())
	if err != nil {
		return toolsFile, err
	}
	return toolsFile, nil
}

// updateLogLevel checks if Toolbox have to update the existing log level set by users.
// stdio doesn't support "debug" and "info" logs.
func updateLogLevel(stdio bool, logLevel string) bool {
	if stdio {
		switch strings.ToUpper(logLevel) {
		case log.Debug, log.Info:
			return true
		default:
			return false
		}
	}
	return false
}

func run(cmd *Command) error {
	if updateLogLevel(cmd.cfg.Stdio, cmd.cfg.LogLevel.String()) {
		cmd.cfg.LogLevel = server.StringLevel(log.Warn)
	}

	ctx, cancel := context.WithCancel(cmd.Context())
	defer cancel()

	// watch for sigterm / sigint signals
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGTERM, syscall.SIGINT)
	go func(sCtx context.Context) {
		var s os.Signal
		select {
		case <-sCtx.Done():
			// this should only happen when the context supplied when testing is canceled
			return
		case s = <-signals:
		}
		switch s {
		case syscall.SIGINT:
			cmd.logger.DebugContext(sCtx, "Received SIGINT signal to shutdown.")
		case syscall.SIGTERM:
			cmd.logger.DebugContext(sCtx, "Sending SIGTERM signal to shutdown.")
		}
		cancel()
	}(ctx)

	// Handle logger separately from config
	switch strings.ToLower(cmd.cfg.LoggingFormat.String()) {
	case "json":
		logger, err := log.NewStructuredLogger(cmd.outStream, cmd.errStream, cmd.cfg.LogLevel.String())
		if err != nil {
			return fmt.Errorf("unable to initialize logger: %w", err)
		}
		cmd.logger = logger
	case "standard":
		logger, err := log.NewStdLogger(cmd.outStream, cmd.errStream, cmd.cfg.LogLevel.String())
		if err != nil {
			return fmt.Errorf("unable to initialize logger: %w", err)
		}
		cmd.logger = logger
	default:
		return fmt.Errorf("logging format invalid")
	}

	ctx = util.WithLogger(ctx, cmd.logger)

	// Set up OpenTelemetry
	otelShutdown, err := telemetry.SetupOTel(ctx, cmd.cfg.Version, cmd.cfg.TelemetryOTLP, cmd.cfg.TelemetryGCP, cmd.cfg.TelemetryServiceName)
	if err != nil {
		errMsg := fmt.Errorf("error setting up OpenTelemetry: %w", err)
		cmd.logger.ErrorContext(ctx, errMsg.Error())
		return errMsg
	}
	defer func() {
		err := otelShutdown(ctx)
		if err != nil {
			errMsg := fmt.Errorf("error shutting down OpenTelemetry: %w", err)
			cmd.logger.ErrorContext(ctx, errMsg.Error())
		}
	}()

	var buf []byte

	if cmd.prebuiltConfig != "" {
		// Make sure --prebuilt and --tools-file flags are mutually exclusive
		if cmd.tools_file != "" {
			errMsg := fmt.Errorf("--prebuilt and --tools-file flags cannot be used simultaneously")
			cmd.logger.ErrorContext(ctx, errMsg.Error())
			return errMsg
		}
		// Use prebuilt tools
		buf, err = prebuiltconfigs.Get(cmd.prebuiltConfig)
		if err != nil {
			cmd.logger.ErrorContext(ctx, err.Error())
			return err
		}
		logMsg := fmt.Sprint("Using prebuilt tool configuration for ", cmd.prebuiltConfig)
		cmd.logger.InfoContext(ctx, logMsg)
		// Append prebuilt.source to Version string for the User Agent
		cmd.cfg.Version += "+prebuilt." + cmd.prebuiltConfig
	} else {
		// Set default value of tools-file flag to tools.yaml
		if cmd.tools_file == "" {
			cmd.tools_file = "tools.yaml"
		}
		// Read tool file contents
		buf, err = os.ReadFile(cmd.tools_file)
		if err != nil {
			errMsg := fmt.Errorf("unable to read tool file at %q: %w", cmd.tools_file, err)
			cmd.logger.ErrorContext(ctx, errMsg.Error())
			return errMsg
		}
	}

	toolsFile, err := parseToolsFile(ctx, buf)
	cmd.cfg.SourceConfigs, cmd.cfg.AuthServiceConfigs, cmd.cfg.ToolConfigs, cmd.cfg.ToolsetConfigs = toolsFile.Sources, toolsFile.AuthServices, toolsFile.Tools, toolsFile.Toolsets
	authSourceConfigs := toolsFile.AuthSources
	if authSourceConfigs != nil {
		cmd.logger.WarnContext(ctx, "`authSources` is deprecated, use `authServices` instead")
		cmd.cfg.AuthServiceConfigs = authSourceConfigs
	}
	if err != nil {
		errMsg := fmt.Errorf("unable to parse tool file at %q: %w", cmd.tools_file, err)
		cmd.logger.ErrorContext(ctx, errMsg.Error())
		return errMsg
	}

	// start server
	s, err := server.NewServer(ctx, cmd.cfg, cmd.logger)
	if err != nil {
		errMsg := fmt.Errorf("toolbox failed to initialize: %w", err)
		cmd.logger.ErrorContext(ctx, errMsg.Error())
		return errMsg
	}

	// run server in background
	srvErr := make(chan error)
	if cmd.cfg.Stdio {
		go func() {
			defer close(srvErr)
			err = s.ServeStdio(ctx, cmd.inStream, cmd.outStream)
			if err != nil {
				srvErr <- err
			}
		}()
	} else {
		err = s.Listen(ctx)
		if err != nil {
			errMsg := fmt.Errorf("toolbox failed to start listener: %w", err)
			cmd.logger.ErrorContext(ctx, errMsg.Error())
			return errMsg
		}
		cmd.logger.InfoContext(ctx, "Server ready to serve!")

		go func() {
			defer close(srvErr)
			err = s.Serve(ctx)
			if err != nil {
				srvErr <- err
			}
		}()
	}

	// wait for either the server to error out or the command's context to be canceled
	select {
	case err := <-srvErr:
		if err != nil {
			errMsg := fmt.Errorf("toolbox crashed with the following error: %w", err)
			cmd.logger.ErrorContext(ctx, errMsg.Error())
			return errMsg
		}
	case <-ctx.Done():
		shutdownContext, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		cmd.logger.WarnContext(shutdownContext, "Shutting down gracefully...")
		err := s.Shutdown(shutdownContext)
		if err == context.DeadlineExceeded {
			return fmt.Errorf("graceful shutdown timed out... forcing exit")
		}
	}

	return nil
}
